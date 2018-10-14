#coding=utf-8
import numpy as np

class SimpleRNN(object):
    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
        self._word_dim = word_dim # size of vocabulary
        self._hidden_dim = hidden_dim # size of hidden state
        self._bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1.0 / self._word_dim), np.sqrt(1.0 / self._word_dim), (self._hidden_dim * self._word_dim))
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
            s[t] = tanh(self.U[:, x[t]] + self.W.dot(x[t]))
            o[t] = self.__softmax(self.W.dot(x[t]))
        return o, s

    def predict(self,x):
        o,s = self.forward_propagation(x)
        return np.argmax(o,axis=1)