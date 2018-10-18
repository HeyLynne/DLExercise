#coding=utf-8
import datetime
import numpy as np
import operator

from math import *

from preprocess import PreProcessor, BaseConfig

class SimpleRNN(object):
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
        self._word_dim = word_dim # size of vocabulary
        self._hidden_dim = hidden_dim # size of hidden state
        self._bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1.0 / self._word_dim), np.sqrt(1.0 / self._word_dim), (self._hidden_dim, self._word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0 / self._hidden_dim), np.sqrt(1.0 / self._hidden_dim), (self._word_dim, self._hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0 / self._hidden_dim), np.sqrt(1.0 / self._hidden_dim), (self._hidden_dim, self._hidden_dim))

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward_propagation(self, x):
        """
        Forwar propagation. The activation function is tanh and softmax
        """
        T = len(x)
        if T == 0:
            raise ValueError("Input sentence length must not be empty!")
        s = np.zeros((T + 1, self._hidden_dim))
        s[-1] = np.zeros(self._hidden_dim)
        o = np.zeros((T, self._word_dim))
        for t in xrange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.__softmax(self.V.dot(s[t]))
        return o, s

    def predict(self,x):
        o, s = self.forward_propagation(x)
        return np.argmax(o,axis=1)

    def __calculate_total_loss(self, x, y):
        """
        Calculate sentences loss.

        Args:
            x: sentences list, input sentence
            y: sentences list, next word sentence
        """
        L = 0
        for i in np.arange(len(y)):
            o,s=self.forward_propagation(x[i])
            correct_word_predictions=o[np.arange(len(y[i])),y[i]]  # prediction of correct words
            L+=-1*np.sum(np.log(correct_word_predictions))
        return L

    def calsulate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.__calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        dldU = np.zeros(self.U.shape)
        dldV = np.zeros(self.V.shape)
        dldW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        for t in np.arange(T)[::-1]:
            dldV = np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt in np.arange(max(0, t - self._bptt_truncate), t + 1)[::-1]:
                dldW += np.outer(delta_t, s[bptt - 1])
                dldU[: ,x[bptt]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt-1] ** 2)
        return [dldU, dldV, dldW]

    def gradient_check(self, x, y, h = 0.001, error_threshold = 0.01):
        bptt_graidents = self.bptt(x, y)
        model_parameters = ["U", "V", "W"]
        for pid, pname in enumerate(model_parameters):
            print pid, pname
            parameter = operator.attrgetter(pname)(self) #get self parameter
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])
            while not it.finished:
                ix = it.multi_index
                print ix
                orinigal_value = parameter[ix]
                parameter[ix] = orinigal_value + h
                gradplus = self.__calculate_total_loss([x], [y])
                parameter[ix] = orinigal_value - h
                gradminus = self.__calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                parameter[ix] = orinigal_value
                backprop_gradient = bptt_graidents[pid][ix]
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error < error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)


if __name__ == "__main__":
    # file_path = "D:\\python script\\deep_learning\\DL_exercise\\DLExercise\\RNN\\data\\reddit.csv"
    # preprocessor = PreProcessor()
    # preprocessor.process_word_index(file_path)
    # X_train = preprocessor.X_train[10]
    # Y_train = preprocessor.Y_train[10]
    # vocabulary_size = BaseConfig.vocabulary_size
    # np.random.seed(10)
    # rnn = SimpleRNN(vocabulary_size)
    # print rnn.bptt(X_train, Y_train)

    verb_size = 100
    np.random.seed(10)
    model = SimpleRNN(verb_size, 10, 1000)
    model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])