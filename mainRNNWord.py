import sys
import os
import time
import numpy as np
from datetime import datetime
from wordLM_RNN import WordRNN

from loader import load_data_word

VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '2000'))
HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
NEPOCH = int(os.environ.get('NEPOCH', '100'))
MODEL_FILE = os.environ.get('MODEL_FILE')
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015-08.csv")


def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("model/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


# Load data
"""
x_train: is a numpy array, every entry is a list(i.e. a sentence)
y_train: similar to x_train
"""
x_train, y_train, word_to_index, index_to_word = load_data_word(INPUT_DATA_FILE, VOCABULARY_SIZE)

print x_train.shape
print y_train.shape

model = WordRNN(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM)
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if MODEL_FILE != None:
    load_model_parameters_theano(MODEL_FILE, model)

train_with_sgd(model, x_train, y_train, nepoch=NEPOCH, learning_rate=LEARNING_RATE)