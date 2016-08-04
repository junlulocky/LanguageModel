import sys
import os
import time
from datetime import datetime
from utils import *
from wordLM_RNN import RNNTheano
import theano as theano
import theano.tensor as T
from utils import *
import operator


class RNNChar:
    def __init__(self, char_dim, hidden_dim=100, bptt_truncate=-1):
        # Assign instance variables
        self.char_dim = char_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters - depends on the incoming connections from previous layer
        U = np.random.uniform(-np.sqrt(1./char_dim), np.sqrt(1./char_dim), (hidden_dim, char_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (char_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        bh = np.zeros((hidden_dim)) # hidden bias
        by = np.zeros((char_dim)) # output bias
        mW, mU, mV = np.zeros_like(W), np.zeros_like(U), np.zeros_like(V)
        mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad

        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.bh = theano.shared(name='bh', value=bh.astype(theano.config.floatX))
        self.by = theano.shared(name='by', value=by.astype(theano.config.floatX))

        self.mW = theano.shared(name='mW', value=mW.astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=mU.astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=mV.astype(theano.config.floatX))
        self.mbh = theano.shared(name='mbh', value=mbh.astype(theano.config.floatX))
        self.mby = theano.shared(name='mby', value=mby.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W, bh, by = self.U, self.V, self.W, self.bh, self.by
        mU, mV, mW, mbh, mby = self.mU, self.mV, self.mW, self.mbh, self.mby
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W, bh, by):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev) + bh)
            o_t = T.nnet.softmax(V.dot(s_t) + by)  # each row representing a word, o is the predicted prob
            return [o_t[0], s_t]  # o_t[0] is in the shape of(char_dim, )
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W, bh, by],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        dbh = T.grad(o_error, bh)
        dby = T.grad(o_error, by)
        mU += dU * dU
        mV += dV * dV
        mW += dW * dW
        mbh += dbh * dbh
        mby += dby * dby

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error, allow_input_downcast=True)
        self.bptt = theano.function([x, y], [dU, dV, dW, dbh, dby])

        # SGD - AdaGrad
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [],
                      updates=[(self.U, self.U - learning_rate * dU / T.sqrt(mU + 1e-8)),
                              (self.V, self.V - learning_rate * dV / T.sqrt(mV + 1e-8)),
                              (self.W, self.W - learning_rate * dW / T.sqrt(mW + 1e-8)),
                              (self.bh, self.bh - learning_rate * dbh / T.sqrt(mbh + 1e-8)),
                              (self.by, self.by - learning_rate * dby / T.sqrt(mby + 1e-8)),
                               (self.mU, mU),
                               (self.mV, mV),
                               (self.mW, mW),
                               (self.mbh, mbh),
                               (self.mby, mby)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([1 for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)



    def sample(self, seed_ix, n=100):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        U, V, W, bh, by = self.U.eval(), self.V.eval(), self.W.eval(), self.bh.eval(), self.by.eval()


        s_t = np.zeros((self.hidden_dim))
        ixes = []
        ixes.append(seed_ix)
        for t in xrange(n):
            s_t = np.tanh(U[:,seed_ix] + np.dot(W, s_t) + bh)
            o_t = np.dot(V, s_t) + by
            o_t = np.exp(o_t) / np.sum(np.exp(o_t))

            seed_ix = np.random.choice(range(self.char_dim), p=o_t)
            ixes.append(seed_ix)
        return ixes


def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)



