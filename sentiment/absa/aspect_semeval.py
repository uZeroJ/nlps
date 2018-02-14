"""
This is an implementation of paper
"Attention-based LSTM for Aspect-level Sentiment Classification" with Keras.
Based on dataset from "SemEval 2014 Task 4".
"""

import os
from time import time
# TODO, Here we need logger!


import numpy as np
from lxml import etree
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import RepeatVector, Dot, Concatenate, Reshape
from keras.activations import softmax
from keras.models import Model, load_model
from keras import regularizers, initializers, optimizers
from keras.layers import Lambda
import keras.backend as K

TEXT_KEY = 'text'
TERM_KEY = 'aspect_terms'
CATEGORY_KEY = 'aspect_categories'

I_TEXT, I_ASPECT, I_POLARITY = 0, 1, 2

# Correspond to settings in paper.
EMBEDDING_DIM = 300
ASPECT_EMBEDDING_DIM = 300
HIDDEN_LAYER_SIZE = 300

# Hyper-parameters for training.
L2_REGULARIZATION = 0.001
MOMENTUM = 0.9
LEARNING_RATE = 0.001
MINI_BATCH_SIZE = 25

RANDOM_UNIFORM = .01

POLARITY_TO_INDEX = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    'conflict': 3
}


def extract_data(data_file='Restaurants_Train_v2.xml'):
    """
    Extract train data from xml file provided buy 'SemEval 2014 Task 4."
    :param file: XML file that contains training data.
    :return: A list of dictionaries of training data with TEXT_KEY, 'aspect
    terms' and 'aspect categories'.
    """
    tree = etree.parse(data_file)
    sents_root = tree.getroot()
    data = []

    def get_content(sent):
        """
        Get all contents from a single 'sentence node', including TEXT_KEY,
        values of 'aspect terms' and 'aspect categories'.
        :param sent: a single xml node of sentence.
        :type: _Element
        :return: A dictionary of contents.
        """
        content = {}
        # We assume that there is must a text node here.
        content[TEXT_KEY] = sent.xpath(TEXT_KEY)[0].text

        terms = sent.xpath('aspectTerms')
        if terms:
            # As there is only one element of 'aspectTerms'.
            # And we only need the first two values, 'aspect' and 'polarity'.
            content[TERM_KEY] = list(map(lambda term: term.values()[:2],
                                         terms[0].iterchildren()))
        else:
            pass

        categories = sent.xpath('aspectCategories')
        if categories:
            content[CATEGORY_KEY] = list(
                map(lambda category: category.values(),
                    categories[0].iterchildren()))
        else:
            pass

        return content

    for sent in sents_root.iterchildren():
        data.append(get_content(sent))

    return data


def check_absent(data):
    """
    Checking absent 'aspect terms' or 'aspect categories'.
    And check if there is sentence missing both 'terms' and 'categories'.
    :param data: dataset with all contents. And the max length of all sentence.
    :type: list of dictionary.
    :return: sentence indices that with absent terms, categories and flag of
    both missing as well as their count separately in tuple.
    :type: tuple of (list, list, boolean)
    """
    exist_both_missing = False
    term_absent_indices = []
    term_absent_cnt = 0
    category_absent_indices = []
    category_absent_cnt = 0
    max_len = 0

    for idx, sent in enumerate(data):
        max_len = max(len(sent[TEXT_KEY]), max_len)

        term_absent = TERM_KEY not in sent.keys()
        category_absent = CATEGORY_KEY not in sent.keys()
        if term_absent and category_absent:
            exist_both_missing = True

        if term_absent:
            term_absent_indices.append(idx)
            term_absent_cnt += 1

        if category_absent:
            category_absent_indices.append(idx)
            category_absent_cnt += 1

    return (term_absent_indices, term_absent_cnt,
            category_absent_indices, category_absent_cnt,
            exist_both_missing, max_len)


def combine_data(data, mess=True, replace_space=True, replace_space_char='_'):
    """
    If `mess` is True, means we would mess all data together.
    Combine text with all aspects related to it, both aspect
    terms and aspect categories. And mess them up.
    But if `mess` is False. we will combined TEXT_KEY and aspect separately
    with 'terms' or 'categories', and return them as tuple.
    And also return the max length of sentence per term or category
    if `mess` is True or separate max length if `mess` is False.
    :param data: all data with TEXT_KEY and lists of 'aspect terms' and
    'categories'.
    :return: all combined data or combined data with 'aspect terms' and
    'categories' separately along with their max length or in all.
    """
    term_data, category_data = [], []
    term_max_len, category_max_len = 0, 0

    # TODO, How do we treat multi-word token as aspect term?
    # 1. take whole as one token an replace space with other mask.
    # 2. split into multiple tokens and average all embeddings.
    # 3. only take one word into consideration.
    # Note for aspect terms, it could contains spaces in the word, so should
    # not use space to split tokenizer, and take all as one token.
    # And also, there are other special characters in the phrase, like '-'.
    # They should be keep.
    for sent in data:
        text = sent[TEXT_KEY]
        is_term_exist = TERM_KEY in sent.keys()
        is_category_exist = CATEGORY_KEY in sent.keys()
        if is_term_exist:
            term_max_len = max(term_max_len, len(sent[TEXT_KEY]))
            for term, polarity in sent[TERM_KEY]:
                if replace_space:
                    term = term.replace(' ', replace_space_char)
                term_data.append([text, term, polarity])
        if is_category_exist:
            category_max_len = max(category_max_len, len(sent[TEXT_KEY]))
            for category, polarity in sent[CATEGORY_KEY]:
                if replace_space:
                    category = category.replace(' ', replace_space_char)
                category_data.append([text, category, polarity])

    # print(len(term_data), len(category_data))

    if mess:
        max_len = max(term_max_len, category_max_len)
        term_data.extend(category_data)
        return term_data, max_len
    else:
        return (term_data, term_max_len), (category_data, category_max_len)


def convert_data(data, max_len=None, with_label=True, extra_data=False):
    """
    Convert data to tuples of (word_vectors, aspect_indices, polarity) to
    word indices sequences and labels to one hot. In order to lookup in
    embedding layer.
    And convert polarity to class identifier, as defined by default in
    polarity to index.

    NOTE: keep in mind to match label and 'text' and 'aspect'!

    :param data: List of data with element of (text, aspect, polarity).
    :param word_vectors: Word Vector lookup table.
    :param with_label: Whether it is training data with label or
        test/customized data without label.
    :return: Arrays contain (word vectors, aspect indices, polarity class
    index), and each of them is a numpy array, along with the word to index
    dictionary.
    :type: numpy array.
    """
    # Set indicator for 'text', 'aspect' and 'polarity(label)'.
    converted_data, lookups = [], []

    texts, aspects, labels = [], [], []
    # TODO, we should count max length here?!
    for d in data:
        texts.append(d[I_TEXT])
        aspects.append(d[I_ASPECT])
        if with_label:
            labels.append(d[I_POLARITY])

    def convert_to_indices(examples, max_len=None, need_tokenizer=False,
                           customized_filter='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        """
        Fit and convert word to indices sequences and word index lookup, and if
        needed, return tokenizer as well.
        :param examples: list of words or sentences.
        :param max_len: the max length of indices sequences.
        :param need_tokenizer: return tokenizer or not.
        :type: boolean
        :return: (indices sequence, word index lookup, <tokenizer>)
        :type: tuple
        """
        tokenizer = Tokenizer(filters=customized_filter)
        tokenizer.fit_on_texts(examples)
        seqs = tokenizer.texts_to_sequences(examples)
        word_idx = tokenizer.word_index
        # TODO, do we need to pad, if yes, 'pre' or 'post'?
        if max_len:
            seqs = pad_sequences(seqs, maxlen=max_len)

        if need_tokenizer:
            return seqs, word_idx, tokenizer
        else:
            return seqs, word_idx

    text_seqs, text_word_idx = convert_to_indices(texts, max_len)
    converted_data.append(np.asarray(text_seqs, dtype='int32'))
    lookups.append(text_word_idx)

    # For aspect term maybe we should not use tokenizer and filter.
    aspects_seqs, aspects_idx = convert_to_indices(
        aspects,
        # TODO, should use less filter.
        customized_filter='#$%&/:;<=>?@[\\]^`{|}~\t\n')
    converted_data.append(np.asarray(aspects_seqs, dtype='int32'))
    lookups.append(aspects_idx)

    if with_label:
        labels_seqs, labels_idx = convert_to_indices(labels)

        # Normalize label sequences as we only need '4' classes and do not need
        # extra class for 'other'.
        labels_arr = np.asarray(labels_seqs, dtype='int') - 1
        labels_one_hot = to_categorical(labels_arr)  # aspects_seqs,
        # [:, np.newaxis],
        converted_data.append(labels_one_hot)
        lookups.append(labels_idx)
        # print(aspects_seqs)

        #    # Preprocessing text without max number of words.
        #    text_tokenizer = Tokenizer()
        #    text_tokenizer.fit_on_texts(texts)
        #    text_seqs = text_tokenizer.texts_to_sequences(texts)
        #    text_word_idx = text_tokenizer.word_index
        #    # Just get indices of words, and does not categorize it as we won't
        #    # multiply one-hot vector in practice as it is computation costly.
        #    # Instead we just lookup with embedding layer.
        #    text_data = pad_sequences(text_seqs, maxlen=max_len)
        #
        #    # Preprocessing aspects.
        #    # The same as word in text, it will be lookup in embedding layer.
        #    aspects_tokenizer = Tokenizer()
        #    aspects_tokenizer.fit_on_texts(aspects)
        #    aspects_seqs = aspects_tokenizer.texts_to_sequences(aspects)
        #    aspects_idx = aspects_tokenizer.word_index
        #
        #    # Processing labels
        #    # Convert labels from words into indices and then to one-hot categorical
        #    # indices.
        #    labels_tokenizer = Tokenizer()
        #    labels_tokenizer.fit_on_texts(labels)
        #    labels_seqs = labels_tokenizer.texts_to_sequences(labels)
        #    labels_idx = labels_tokenizer.

    return converted_data, lookups


def load_w2v(idxes, emb_file, save_to_file=None):
    """
    Load pre-trained embedding and match words in training data to form a
    small set of word embedding matrix with OOV with all '0's.
    NOTE: Keras tokenizer.word_index start from 1, in order to use '0'
    padding in pad_sequence and mask_zero in embedding layer and following
    layer.
    :param idxes: the word loopup dictionary of word indices.
    :param emb_file: pre-trained embedding file.
    :return: word embedding matrix fit for the training data.
    """
    # Only need the lookup for 'text'.
    idx = idxes[I_TEXT]

    # Initial word embedding matrix with all '0's.
    # TODO, here we could set embedding dimesion automatically.
    emb_matrix = np.zeros((len(idx) + 1, EMBEDDING_DIM))

    # Timing it.
    start_time = time()
    with open(emb_file) as emb:
        for line in emb:
            pieces = line.strip().split()
            word, coef = pieces[0].strip(), pieces[1:]
            begin_idx = 0
            for elem_idx, elem in enumerate(coef):
                # In case there is space in the word,
                # continuously test if the string could be interpret as float,
                # if yes, it means this piece element is the beginning of the
                # coefficient and if no, then append to word as part of the
                # token.
                try:
                    # Test if an element in coefficient is an actual
                    # coefficient of a part of key token.
                    float(elem)
                    # Record begin index of actual coefficient.
                    begin_idx = elem_idx + 1
                    # Only break when we find the begin index of actual
                    # coefficient.
                    break
                except Exception as e:
                    word += elem
                    # print(e)
                    # TODO, we could record the trail and error in log.
                    # print("Filed to load record with word: '{}' and "
                    #       "coefficient: {}".format(word, coef))
                    # print(word)
            coef = np.asarray(pieces[begin_idx:], dtype=np.float32)

            if word in idx.keys():
                # Lookup the indices(index) of word and set the corresponding
                # vector to the one in pre-trained embedding matrix.
                emb_matrix[idx[word]] = coef
        print('Loaded word embedding matrix within {}'.format(
            time() - start_time))

    # Save loaded subset of word embedding into files.
    if save_to_file:
        np.save(save_to_file, emb_matrix)

    return emb_matrix


def build_net(data, max_len, w2is, atae=True, extra_outputs=True,
              emb_mtrx_file=None, save_to_file=None):
    """
    Build ATAE-LSTM mentioned in paper 'Attention-based LSTM for Aspect-level
    Sentiment Classification', with uniform randomly initialized aspect
    embedding and word embedding subset according training data and given
    pre-trained embedding file.
    Adapt 'inter' attention before do multiple classes classification by
    softmax, which introduce aspect-level attention as part of the encoding
    of source sentence.`

    :param data: Indices of training data including (sentences, aspect,
        polarity(one-hot label))
    :param max_len: the max length of sentence as it has been padding with
    '0's and need to set for the input shape with mini-batch.
    :param w2is: Index lookup table of components above.
    :param atae: If 'False' then only use 'AE'.
    :param extra_outputs: return extra outputs like attention weights,
        aspect embeddings or so.
    :param emb_mtrx_file: Pre-saved embedding matrix corresponding to
        training data and given pre-trained embedding. If 'None' is set,
        then reload from embedding file.
    :param save_to_file: File path to save model, if 'None' is set, then its
    a one way training.
    :return: Training loss and accuracy for all classes?
    """
    # TODO, max length should be fixed.
    sents, aspects, labels = data
    sents_idx, aspects_idx, _ = w2is
    emb_mtrx = np.load(emb_mtrx_file)

    # Input of sentences.
    sents_tensor_input = Input(shape=(sents.shape[1],), dtype='int32')
    # Do not retrain embedding of sentences.
    sents_tensor = Embedding(len(sents_idx) + 1,
                             # EMBEDDING_DIM
                             emb_mtrx.shape[1],
                             weights=[emb_mtrx],
                             input_length=max_len,
                             trainable=False)(sents_tensor_input)

    # Input of aspect
    # As we use ATAE-LSTM, aspect embedding need to be concated to each time
    # steps in sentences.
    # Aspect is a single index of integer.
    aspects_tensor_input = Input(shape=(1,), dtype='int32')
    # Randomly initialize aspect embedding.
    aspects_emb_initializer = initializers.RandomUniform(minval=-RANDOM_UNIFORM,
                                                         maxval=RANDOM_UNIFORM)
    aspects_emb_layer = Embedding(len(aspects_idx) + 1,
                                  ASPECT_EMBEDDING_DIM,
                                  embeddings_initializer=aspects_emb_initializer,
                                  trainable=True,
                                  name='asp_emb_layer')
    # In order to get embedding weights.
    # aspects_emb_matrix = Lambda(lambda x: x, name='asp_emb_weight')(
    #     aspects_emb_layer.weights)
    aspects_emb = aspects_emb_layer(aspects_tensor_input)
    # Here, before repeat we need reshape aspect_tensor act as 'squeeze' with
    # the dimension with '1', say Reshape((10, ), input_shape=(1, 10))(...)
    # then got keras tensor with shape of (10,), which will then feed into
    # `RepeatVector`.
    aspects_tensor = Reshape((ASPECT_EMBEDDING_DIM,))(aspects_emb)
    # Repeat aspects tensor in order to correspond to the time step of
    # sentences, with shape of (max_len, ASPECT_EMBEDDNING_DIM).
    # TODO, could use Timedistributed?
    aspects_tensor = RepeatVector(max_len)(aspects_tensor)

    lstm_input = Concatenate()([sents_tensor, aspects_tensor])

    if atae:
        lstm_output = LSTM(HIDDEN_LAYER_SIZE, return_sequences=True)(lstm_input)
        # Attention with concatenation of sequential output of LSTM and
        # aspect embedding.
        attention_input = Concatenate()([lstm_output, aspects_tensor])
        attention_score = Dense(EMBEDDING_DIM + ASPECT_EMBEDDING_DIM,
                                use_bias=False,
                                name='attention_score_1')(attention_input)
        # We need an extra `Dense/Activation` layer here for axis related
        # softmax with should be align on time step instead the last axis.
        attention_weight = Dense(1, use_bias=False,
                                 name='attention_score_2')(attention_score)
        attention_weight = Lambda(lambda x: softmax(x, axis=1))(
            attention_weight, name='attention_weights')

        # permuted_weight = Permute((2, 1))(attention_weight)
        # attention_represent = Multiply(name='r')([lstm_output, permuted_weight])
        # attention_represent = Multiply(name='r')([lstm_output, attention_weight])
        attention_represent = Dot(axes=1, name='r')([lstm_output,
                                                     attention_weight])
        attention_represent = Reshape((EMBEDDING_DIM,))(attention_represent)

        last_hidden = Lambda(lambda tensor: tensor[:, -1, :])(lstm_output)
        final_represent = Concatenate(name='final_concatenate')([
            attention_represent, last_hidden])
        final_represent = Dense(EMBEDDING_DIM, activation='tanh',
                                use_bias=False, name='final_representation')(
            final_represent)
        model_output = Dense(labels.shape[1],
                             activation='softmax',
                             activity_regularizer=regularizers.l2(
                                 L2_REGULARIZATION),
                             name='ATAE_LSTM_output')(final_represent)

        # outs = [model_output]
        # if extra_outputs:
        #     outs.append(attention_weight)
        #     TODO, get from model outside
        # outs.append(aspects_emb_matrix)
        # print(outs)
    else:
        lstm_output = LSTM(HIDDEN_LAYER_SIZE,
                           return_sequences=False)(lstm_input)
        model_output = Dense(labels.shape[1],
                             activation='softmax',
                             name='Simple_AE_LSTM_ouptut')(lstm_output)
        # outs = [model_output]

    model = Model(inputs=[sents_tensor_input,
                          aspects_tensor_input],
                  outputs=model_output)

    if save_to_file:
        model.save(save_to_file)

    return model


def train(data, model, model_optimizer=None, metrics=None, valid_ratio=0.1,
          epoch=10, mini_batch=25, save_to_file=None):
    """
    :param data: Training data in tuples of lists with form of (sentences,
        aspect word, polarity).
    :param model: Predefined model generated by `build_net`, if None,
        then if will be build with default values.
    :param optimizer: Optimizer used to train/compile model. Default is
        Adagrad with learning rate as '0.001'.
    :param metrics: Metrics are interested in list. If not set then default
        is ['accuracy']
    :return: None
    """

    if not model and not data:
        print('Please passed in data and model!')
        return

    if not metrics:
        metrics = ['accuracy']

    if not model_optimizer:
        model_optimizer = optimizers.Adagrad(lr=0.001)

    print("Training Model ...")
    print(model.summary())
    # print('\t\twith data as')
    # print('\t\t{}'.format(check_absent(data)))
    print('\t\twith hyper-parametes as')
    print('\t\t\tMini-Batch : {}'.format(mini_batch))
    print('\t\t\tEpoch : {}'.format(epoch))

    model.compile(model_optimizer, 'categorical_crossentropy', metrics=metrics)
    model.fit([seqs_data[I_TEXT], seqs_data[I_ASPECT]], seqs_data[I_POLARITY],
              mini_batch, epochs=epoch, validation_split=valid_ratio)

    if save_to_file:
        model.save(save_to_file)


def train_dev_split(data, ratio=0.8, seed=42):
    """
    Function to split train and dev set with given ratio.
    :param data: whole dataset.
    :param ratio: percentage that training data occupied.
    :return: tuple of list of (training, dev), and each of them should be
        formed as (sentences, aspect word, polarity)
    """
    np.random.seed(42)
    sents, aspects, labels = data[I_TEXT], data[I_ASPECT], data[I_POLARITY]
    idx = np.arange(sents.shape[0])
    np.random.shuffle(idx)

    sents = sents[idx]
    aspects = aspects[idx]
    labels = labels[idx]

    # Calculate split boundary.
    bnd = int(len(idx) * ratio)

    train_set = [sents[:bnd], aspects[:bnd], labels[:bnd]]
    dev_set = [sents[bnd:], aspects[bnd:], labels[bnd:]]

    return train_set, dev_set


def predict(data, lookup, max_len, model=None, save_to_file=None,
            extra_output=True):
    """
    Predict with given data and model or load model from saved pre-trained
    model in file.
    :param data: data in tuple or list (sentence, aspect)
    :param w2is: index to lookup for predictions.
    :param max_len: length to padding to.
    :param model: pre-trained model, if not set loaded from file,
        and if file for model is also not set, return with error.
    :param save_to_file: file saved with model.
    :return: prediction
    """
    # Omit word index lookups.
    converted_data, _ = convert_data(data, max_len, with_label=False)
    # print(converted_data)
    if not model:
        if save_to_file:
            model = load_model(save_to_file,
                               custom_objects={'softmax': softmax})
        else:
            # TODO, should raise exception?
            raise ValueError('Please pass in model instance or '
                             'the path of file model saved to.')
    pred_vec = model.predict([converted_data[I_TEXT],
                              converted_data[I_ASPECT]])
    pred_idx = np.argmax(pred_vec, axis=1)
    func_get_label = np.vectorize(lambda p: lookup.get(p))
    # print(pred_idx, func_get_label(pred_idx), lookup.get(0))
    # Need to add '1' for keras labels start from '0'.
    pred = func_get_label(pred_idx + 1)

    # if extra_output:
    #     model.layers

    return pred


def get_layer(model, layer_name):
    """
    Get layer from model by name or index.
    :param layer_name: the name or index of layer.
    :return: layer instance extract from model.
    """
    if isinstance(layer_name, int):
        return model.layers[layer_name]
    elif isinstance(layer_name, str):
        return model.get_layer(layer_name)
    else:
        raise ValueError('The layer name should only be `int` or `str`.')


def get_aspect_embeddings(model, layer_name, save_to_file=None):
    """
    Get aspect embedding from specific layer with given name.
    :param model: the pre-trained model, if not set, reload form saved model
        file. If it also failed to load model from file, 'ValueError' will be thrown.
    :param layer_name: the name or index of embedding layer, or ValueError
        will be thrown.
    :param save_to_file: file saved pre-trained model, load model if model is 'None'.
    :return: tensor of apsect embeddings.
    """
    if not model:
        if not save_to_file:
            raise ValueError('No model found from parameter or file!')
        else:
            model = load_model(save_to_file)
    # Get embeddings of aspect words.
    emb_layer = get_layer(model, layer_name)
    return K.eval(emb_layer.embeddings)


def get_attention_weighs(data, att_layer_name, input_layers_names: list,
                         model=None, save_to_file=None):
    """
    Get attention weights(intermediate) from specific layer with given layer
        name and input layers.
    :param data: data to attendant to.
    :param model: the pre-trained model, if not set, reload form saved model
        file. If it also failed to load model from file, 'ValueError' will be thrown.
    :param att_layer_name: the name or index of embedding layer, or ValueError
        will be thrown.
    :param input_layers: the name or index list of all input layer in order.
    :param save_to_file: file saved pre-trained model, load model if model is 'None'.
    :return: tensor of attention indices.
    """
    if not model:
        if not save_to_file:
            raise ValueError('No model found from parameter or file!')
        else:
            model = load_model(save_to_file,
                               custom_objects={'softmax': softmax})

    # Must be sure input layers are in order.
    att_layer = get_layer(model, att_layer_name)

    input_layers = []
    for layer_name in input_layers_names:
        layer = get_layer(model, layer_name)
        if layer:
            input_layers.append(layer.input)

    get_attention_weights = K.function(input_layers, [att_layer.output])
    weights = get_attention_weights([data[I_TEXT], data[I_ASPECT]])[0]
    # print(weights.shape)

    return weights


def plot_attention_weight(weights, focus_len):
    """
    Plot attention weights within the focus length.
    :param weights: attention weights.
    :param focus_len: the length to focus to, usually the length of sentences.
    :return: None
    """
    # score_file = os.path.join(RAW_DATA_FILE_BASE, 'intermeidate_score')
    # np.save(score_file, weights)

    # score_input = Input(shape=(term_max_len, 600))
    # get_weights = Dense(1, use_bias=False)(score_input)
    # get_weights = Activation('softmax', axis=1)(get_weights)
    # get_weights = Lambda(lambda x: tf.nn.softmax())
    # from keras.activations import softmax

    # # # get_weights = Lambda(lambda x: softmax(x, axis=1))(get_weights)
    # # # score_model = Model(score_input, get_weights)
    # # # print(score_model.summary())
    # #
    # # score_model.compile(optimizer='adam', loss='categorical_crossentropy')
    # weight_result = score_model.predict(weights)
    # print(weight_result[0].shape)

    # begin_idx = len(converted_data[I_TEXT][0])
    # print(begin_idx)
    import matplotlib.pyplot as plt

    # hist, bins = np.histogram(weight_result[0].reshape((1, -1)))
    # We have to remember the length of input sentences in order to align the
    # attention weights.
    # plt.imshow(weight_result[0][-20:].reshape((1, -1)), cmap="plasma",
    #            aspect="auto", extent=[0, 20, 0, 1])

    # TODO, Here is 'pre pad', so its '-focus_len' for the actual token.
    attentions = weights.reshape((1, -1))[:, -focus_len:]
    print(attentions.shape)
    plt.imshow(attentions, cmap='plasma',
               aspect='auto', extent=[0, focus_len, 0, 1])
    # plt.grid(True)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    RAW_DATA_FILE_BASE = '/Users/jiazhen/datasets/SemEval' \
                         '/SemEval_2014_task4/ABSA_v2'
    RES_RAW_DATA_FILE = os.path.join(RAW_DATA_FILE_BASE,
                                     'Restaurants_Train_v2.xml')
    LAP_RAW_DATA_FILE = os.path.join(RAW_DATA_FILE_BASE, 'Laptop_Train_v2.xml')

    WORD_EMB_BASE = '/Users/jiazhen/datasets'
    WORD_EMB_FILE = os.path.join(WORD_EMB_BASE, 'glove.840B.300d.txt')

    SAVED_EMB_FILE = os.path.join(RAW_DATA_FILE_BASE, 'glove_res_emb.npy')
    SAVED_MDL_FILE = os.path.join(RAW_DATA_FILE_BASE, 'atae_model.keras')

    res_data = extract_data(RES_RAW_DATA_FILE)

    # print(res_data[7])
    check_absent(res_data)
    (term_data, term_max_len), _ = combine_data(res_data, mess=False)
    # print(term_data[7])

    # No padding here according to the paper.
    # Need padding for mini-batch.
    seqs_data, w2is = convert_data(term_data, max_len=term_max_len)

    # emb_matrix = load_w2v(w2is, WORD_EMB_FILE, SAVED_EMB_FILE)
    # print(emb_matrix[1])

    # print(len(seqs_data))
    # print(seqs_data[0].shape, seqs_data[1].shape, seqs_data[2].shape)
    # print(seqs_data[1])
    # for i, d in enumerate(seqs_data[1]):
    #     if len(d) > 1:
    #         print(i, d)
    #         print(term_data[i][I_ASPECT])
    # print('raw data', res_data[92]['aspect_terms'])
    # print(type(seqs_data[1][0][0]))
    # print(type(seqs_data[2][0][0]))
    # print(w2is[0])

    # reloaded_emb = np.load(SAVED_EMB_FILE)
    # print(reloaded_emb[1])

    # Train model.
    # model = build_net(seqs_data, term_max_len, w2is,
    #                   atae=True, extra_outputs=True,
    #                   emb_mtrx_file=SAVED_EMB_FILE,
    #                   save_to_file=SAVED_MDL_FILE + '2')
    # train(seqs_data, model, epoch=3)

    label_lookup = {idx: polarity
                    for polarity, idx in w2is[I_POLARITY].items()}
    # print(label_lookup)
    customized_data = [['The food is really delicious but '
                        'I hate the service', 'food'],
                       ['The food is really delicious but '
                        'I hate the service', 'serivce'],
                       ['I have to say there is no on could be faster than '
                        'him, but he need to take care of his bad motion as '
                        'a bar attendant, which will impact his serivce.',
                        'serivce']]
    pred = predict(customized_data, label_lookup, term_max_len,
                   save_to_file=SAVED_MDL_FILE + '2')
    print(pred)

    # Get attention weights for sentences.
    converted_data, _ = convert_data(customized_data,
                                     term_max_len,
                                     with_label=False)

    weights = get_attention_weighs(converted_data,
                                   att_layer_name='attention_weight',
                                   # att_layer_name='attention_weights',
                                   input_layers_names=[2, 0],
                                   save_to_file=SAVED_MDL_FILE + '2')
    # print(weights[0])
    print(len(customized_data[0][I_TEXT].split()))
    focus_len = len(customized_ata[0][I_TEXT].split())
    plot_attention_weight(weights[0], focus_len=focus_len)
    # for weight in weights:
    #     print(weight.shape)

    # TODO, Use gemsim to visualize aspect word embeddings.
