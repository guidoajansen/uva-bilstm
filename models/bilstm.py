"""Load packages"""

import numpy as np
import tensorflow as tf
import os.path

from helpers.preprocessing import readfile, addCharInformation, createBatches, createMatrices, iterate_minibatches, padding


"""Initialise class"""

class CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER

    """Load data and add character information"""
    def loadData(self):
        self.trainSentences = readfile("train.txt")
        self.devSentences = readfile("dev.txt")
        self.testSentences = readfile("test.txt")

    def addCharInfo(self):
        # format: [['EU', ['E', 'U'], 'B-ORG\n'], ...]
        self.trainSentences = addCharInformation(self.trainSentences)
        self.devSentences = addCharInformation(self.devSentences)
        self.testSentences = addCharInformation(self.testSentences)

    def embed(self):
        """Create word- and character-level embeddings"""

        labelSet = set()
        words = {}

        # unique words and labels in data
        for dataset in [self.trainSentences, self.devSentences, self.testSentences]:
            for sentence in dataset:
                for token, char, label in sentence:
                    # token ... token, char ... list of chars, label ... BIO labels
                    labelSet.add(label)
                    words[token.lower()] = True

        # mapping for labels
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

        # mapping for token cases
        case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  # identity matrix used

        # read GLoVE word embeddings
        word2Idx = {}
        self.wordEmbeddings = []

        directory = os.path.join(os.path.dirname(__file__), os.pardir + '/embeddings/glove/')
        fEmbeddings = open(directory + 'glove.6B.300d.txt', encoding = 'utf-8')

        # loop through each word in embeddings
        for line in fEmbeddings:
            split = line.strip().split(" ")
            word = split[0]  # embedding word entry

            if len(word2Idx) == 0:  # add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
                self.wordEmbeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
                self.wordEmbeddings.append(vector)

            if split[0].lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                self.wordEmbeddings.append(vector)  # word embedding vector
                word2Idx[split[0]] = len(word2Idx)  # corresponding word dict

        self.wordEmbeddings = np.array(self.wordEmbeddings)

        # dictionary of all possible characters
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
            self.char2Idx[c] = len(self.char2Idx)

        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        self.train_set = padding(createMatrices(self.trainSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        self.dev_set = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        self.test_set = padding(createMatrices(self.testSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))

        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

    def createBatches(self):
        """Create batches"""
        self.train_batch, self.train_batch_len = createBatches(self.train_set)
        self.dev_batch, self.dev_batch_len = createBatches(self.dev_set)
        self.test_batch, self.test_batch_len = createBatches(self.test_set)

    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def buildModel(self):
        """Reset graph"""
        tf.reset_default_graph()

        """Character input"""
        char_input = tf.placeholder(dtype=tf.int32, shape=(None, 52), name="character_input")
        char_embed = tf.get_variable(name="char_embed", shape=[len(self.char2Idx), 30])
        chars = tf.nn.embedding_lookup(params=char_embed, ids=char_input)
        chars = tf.layers.dropout(inputs=chars, rate=self.dropout)

        """CNN"""
        chars = tf.layers.conv1d(inputs=chars, filters=30, kernel_size=self.conv_size, strides=1, padding="same", activation="tanh", name="convolution")
        chars = tf.layers.max_pooling1d(inputs=chars, pool_size=52, strides=1, name="maxpool")
        chars = tf.contrib.layers.flatten(inputs=chars)
        chars = tf.layers.dropout(inputs=chars, rate=self.dropout)

        """Word input"""
        word_input =  tf.placeholder(dtype=tf.int32, shape=(None,), name="word_input")
        word_embed = tf.get_variable(name="word_embed", shape=[self.wordEmbeddings.shape[0], self.wordEmbeddings.shape[1]])
        words = tf.nn.embedding_lookup(params=word_embed, ids=word_input)

        """Case input"""
        case_input = tf.placeholder(dtype=tf.int32, shape=(None,), name="case_input")
        case_embed = tf.get_variable(name="case_embed", shape=[self.caseEmbeddings.shape[0], self.caseEmbeddings.shape[1]])
        casing = tf.nn.embedding_lookup(params=case_embed, ids=case_input)

        """Concat"""
        concat = tf.concat(values=[chars, words, casing], axis=-1)

        """BLSTM"""
        forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_state_size, forget_bias=1.0)
        forward = tf.nn.rnn_cell.DropoutWrapper(cell = forward, input_keep_prob=self.dropout, state_keep_prob=self.dropout_recurrent)
        backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_state_size, forget_bias=1.0)
        backward = tf.nn.rnn_cell.DropoutWrapper(cell = backward, input_keep_prob=self.dropout, state_keep_prob=self.dropout_recurrent)

        blstm = tf.nn.static_bidirectional_rnn(cell_fw=forward, cell_bw=backward, inputs=(100, 338), dtype=tf.float32)

        # output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'),name="Softmax_layer")(output)
        #
        # # set up model
        # self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        #
        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)
        #
        # self.init_weights = self.model.get_weights()
        #
        # plot_model(self.model, to_file='model.png')
        #
        # print("Model built. Saved model.png\n")
