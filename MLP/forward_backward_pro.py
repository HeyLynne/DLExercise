#coding=utf-8
import numpy as np

class NuralNetwork(object):
    def __init__(self, input, hidden, output):
        self._input = input + 1
        self._hidden = hidden
        self._output = output
        self._ai = [1.0] * self._input
        self._ah = [1.0] * self._hidden
        self._ao = [1.0] * self._output
        self._wi = np.random.randn(self._input, self._hidden)
        self._wo = np.random.randn(self._hidden, self._output)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(x))

    def _dsigmoid(self, y):
        return y * (1.0 - y)

    def feed_propagation(self, inputs):
        """
        Feed forward nerual network

        Args:
            inputs: input matrix
        """
        if len(inputs) != self._input - 1:
            raise ValueError("Input matrix does not fit the network!")
        for i in xrange(len(inputs)):
            self._ai[i] = inputs[i]
        for j in xrange(self._hidden):
            sum = 0.0
            for i in xrange(self._input):
                sum += self._wi[i][j] * self._ai[i]
            self._ah[j] = self._sigmoid(sum)
        for j in xrange(self._output):
            sum = 0.0
            for i in xrange(self._hidden):
                sum += self._wo[i][j] * self._ah[i]
            self._ao[j] = self._sigmoid(sum)
        return self._ao[:]

    def back_propagation(self, targets, N):
        """
        Back propagation algorithm.

        Args:
            targets: target number, lists
        """
        if len(targets) != self._output:
            raise ValueError("Target does not match the output matrix!")
        # Caluculate the derivative of each layer's input
        out_delta = [0.0] * self._output
        print out_delta
        for i in xrange(self._output):
            out_delta[i] = -(targets[i] - self._ao[i]) * self._dsigmoid(self._ao[i])
        hidden_delta = [0.0] * self._hidden
        for i in xrange(self._hidden):
            error = 0.0
            for j in xrange(self._output):
                error += out_delta[j] * self._wo[i][j]
            hidden_delta[i] = self._dsigmoid(self._ah[i]) * error
        # Update the parameter of each layer's weight
        for j in range(self._hidden):
            for k in range(self._output):
                change = out_delta[k] * self._ah[j]
                self._wo[j][k] -= N * change
        for i in range(self._input):
            for j in range(self._hidden):
                change = hidden_delta[j] * self._ai[i]
                self._wi[i][j] -= N * change
        # loss function is MSE
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self._ao[k]) ** 2
        return error

    def train(self, patterns, iterations = 3000, N = 0.0002):
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                target = []
                target.append(targets)
                self.feed_propagation(inputs)
                error = self.back_propagation(target, N)
            if i % 500 == 0:
                print ("error %-.5f" % error)

    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions

def demo():
    def load_data():
        data = np.loadtxt("pima-indians-diabetes.data.csv", delimiter = ',')
        y = data[:,8]
        x = data[:, 0 : 7]
        x -= x.min()
        x /= x.max()
        out = []
        for i in range(data.shape[0]):
            fart = list((data[i, :].tolist(),
                         y[i].tolist()))
            out.append(fart)

        return out
    X = load_data()

    NN = NuralNetwork(9, 10, 1)

    NN.train(X)

    NN.test(X)

if __name__ == "__main__":
    demo()