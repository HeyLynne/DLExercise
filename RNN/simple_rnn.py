#coding=utf-8
import datetime
import numpy as np

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
        L = 0
        for i in np.arange(len(y)):
            o,s=self.forward_propagation(x[i])
            correct_word_predictions=o[np.arange(len(y[i])),y[i]]  # prediction of correct words
            print correct_word_predictions
            L+=-1*np.sum(np.log(correct_word_predictions))
        return L

    def calsulate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.__calculate_total_loss(x, y) / N

if __name__ == "__main__":
    file_path = "D:\\python script\\deep_learning\\DL_exercise\\DLExercise\\RNN\\data\\reddit.csv"
    preprocessor = PreProcessor()
    preprocessor.process_word_index(file_path)
    X_train = preprocessor.X_train[:10]
    Y_train = preprocessor.Y_train[:10]
    vocabulary_size = BaseConfig.vocabulary_size
    np.random.seed(10)
    rnn = SimpleRNN(vocabulary_size)
    print rnn.calsulate_loss(X_train, Y_train)