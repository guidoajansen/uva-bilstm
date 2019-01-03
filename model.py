"""Load packages"""

import importlib

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
def loadModel():
    """Reload the model"""

    importlib.reload(bilstm)
    cnn_blstm = bilstm.CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)

    cnn_blstm.loadData()
    cnn_blstm.addCharInfo()
    cnn_blstm.embed()
    cnn_blstm.createBatches()
    cnn_blstm.buildModel()
    # cnn_blstm.train()
    # cnn_blstm.writeToFile()

    return cnn_blstm

MODEL = loadModel()
