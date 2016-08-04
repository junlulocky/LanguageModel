import time
from utils import *

from charLM_RNN import RNNChar
from loader import load_data_char


# hyperparameters
hyper = {'hidden_size':100,
         'seq_length':10,
         'learning_rate':1e-1}


def train_with_sgd(model, x_train, y_train, learning_rate=0.0001, nepoch=1, evaluate_loss_after=1000, ix_to_char=None):

    iter, p = 0, 0
    epoch = 0
    # We keep track of the losses so we can plot them later
    losses = []
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+hyper['seq_length'] >= len(x_train) or iter==0:
            hprev = np.zeros((hyper['hidden_size'],1)) # reset RNN memory
            p = 0 # go from start of data
            epoch += 1
        inputs = [ix for ix in x_train[p:p+hyper['seq_length']]]
        targets = [ix for ix in y_train[p:p+hyper['seq_length']]]

        #for epoch in range(nepoch): Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            sample_ix = model.sample(inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print '----\n %s \n----' % (txt, )
            loss = model.calculate_loss([inputs], [targets])
            losses.append((epoch, loss))
            print "Loss after epoch=%d: %f" % (epoch, loss)
            # Adjust the learning rate if loss increases
            #if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                #learning_rate = learning_rate * 0.5
                #print "Setting learning rate to %f" % learning_rate
            #sys.stdout.flush()
        model.sgd_step(inputs, targets, learning_rate)

        p += hyper['seq_length'] # move data pointer
        iter += 1 # iteration counter



# load data
x_train, y_train, char_to_ix, ix_to_char, vocab_size = load_data_char()

p=0
inputs = [ix for ix in x_train[p:p+hyper['seq_length']]]
targets = [ix for ix in y_train[p:p+hyper['seq_length']]]

model = RNNChar(char_dim=vocab_size, hidden_dim=hyper['hidden_size'])
t1 = time.time()
# initial step
model.sgd_step(inputs, targets, hyper['learning_rate'])
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
train_with_sgd(model, x_train, y_train, nepoch=None, learning_rate=hyper['learning_rate'], ix_to_char=ix_to_char)
