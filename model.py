"""Load packages"""

import numpy as np
import tensorflow as tf

# from keras.models import Model, load_model
# from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
#     Flatten, concatenate
# from keras.utils import plot_model
# from keras.initializers import RandomUniform
# from keras.optimizers import SGD, Nadam

"""Load external scripts"""
from validation import compute_f1
import models.bilstm as bilstm

"""Set parameters"""
EPOCHS = 30               # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 200     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = 'nadam'       # paper uses SGD(lr=self.learning_rate), Nadam() recommended

"""Construct and run model"""
cnn_blstm = bilstm.CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
cnn_blstm.loadData()
cnn_blstm.addCharInfo()
cnn_blstm.embed()
# cnn_blstm.createBatches()
# cnn_blstm.buildModel()
# cnn_blstm.train()
# cnn_blstm.writeToFile()
