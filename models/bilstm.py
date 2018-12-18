"""Initialise class"""

import numpy as np
import os.path

# from models.preprocessing import readfile, createBatches, createMatrices, iterate_minibatches, addCharInformation, padding
from models.preprocessing import readfile, addCharInformation


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

        #     if len(word2Idx) == 0:  # add padding+unknown
        #         word2Idx["PADDING_TOKEN"] = len(word2Idx)
        #         vector = np.zeros(len(split) - 1)  # zero vector for 'PADDING' word
        #         self.wordEmbeddings.append(vector)
        #
        #         word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        #         vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        #         self.wordEmbeddings.append(vector)
        #
        #     if split[0].lower() in words:
        #         vector = np.array([float(num) for num in split[1:]])
        #         self.wordEmbeddings.append(vector)  # word embedding vector
        #         word2Idx[split[0]] = len(word2Idx)  # corresponding word dict
        #
        # self.wordEmbeddings = np.array(self.wordEmbeddings)
        #
        # # dictionary of all possible characters
        # self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        # for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
        #     self.char2Idx[c] = len(self.char2Idx)
        #
        # # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        # self.train_set = padding(createMatrices(self.trainSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        # self.dev_set = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        # self.test_set = padding(createMatrices(self.testSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        #
        # self.idx2Label = {v: k for k, v in self.label2Idx.items()}
