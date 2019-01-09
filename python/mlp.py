# https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers=[2,3,1], lrate=0.5):
        self.lrate = lrate
        self.layers = layers
        #parameters
        self.inputSize = 2
        self.hiddenSize = 3
        self.outputSize = 1
    
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
        self.w=[]
        for i in range(len(layers)-1):
            self.w.append(np.random.randn(layers[i], layers[i+1]))

    def forward(self, X):
        #forward propagation through our network
        z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(z) # activation function
        z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(z3) # final activation function
        return o
        # generalize
        self.output=[]
        last = X
        self.output.append(last)
        for i in range(len(self.layers)-1):
            mid = np.dot(last, self.w[i])
            last = self.sigmoid(mid)
            self.output.append(last)
        return o 

    def backward(self, X, y, o):
        # backward propgate through the network
        o_error = o - y # error in output
        o_delta = o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        z2_error = np.dot(o_delta, self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        z2_delta = z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W2 += -self.lrate * np.dot(self.z2.T, o_delta) # adjusting second set (hidden --> output) weights
        self.W1 += -self.lrate * np.dot(X.T, z2_delta) # adjusting first set (input --> hidden) weights
        return
        # generalize
        delta = np.array()
        for i in range(len(self.layers)-2, -1, -1):
            pred = self.output[i+1]
            if i == len(self.layers) - 2:
                error = pred - y
            else:
                error = np.dot(delta, self.w[i+1].T)
            delta = error * self.sigmoidPrime(pred)
            grad = np.dot(self.output[i].T, delta)
            self.w[i] += -self.lrate * grad
        
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def loss(self, y, p):
        return np.mean(np.square(y - p))
        
    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        v = self.sigmoid(s)
        return v * (1 - v)


def main():
    # X = (hours sleeping, hours studying), y = score on test
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)
    
    # scale units
    X = X/np.amax(X, axis=0) # maximum of X array
    y = y/100 # max test score is 100
    nn = NeuralNetwork()
    for i in range(1000): # trains the NN 1,000 times
        if i % 10 == 0:
            p=nn.forward(X)
            l=nn.loss(y,p)
            print("Iteration %i: loss %f" % (i, l))
        nn.train(X, y)

