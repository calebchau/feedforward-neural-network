import numpy as np
import random

class NeuralNetwork:
    def __init__(self, sizes):
        '''
        Takes in a list of sizes with each entry corresponding to the number of
        neurons in that particular layer of the neural network, e.g., sizes =
        [784, 30, 10] would create a neural network with a 784 node input layer,
        a 30 node hidden layer, and a 10 node output layer. The weights are
        initialized as j x k matrices where j refers to the size of the layer on
        the right side of the connection and k refers to the size of the layer
        on the left. The biases are initialized for all layers except the input
        layer. Both are initialized randomly with a univariate Gaussian
        distribution with mean 0 and variance 1.
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(j, k)
                        for k, j in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]

    def train(self, training_data, num_epochs, mini_batch_size, eta,
              test_data=None):
        '''
        Trains the neural network using stochastic gradient descent and randomly
        shuffled mini batches of training samples. The training data is expected
        as a zip object of 2-tuples which represent the inputs and desired
        outputs. The other parameters may be tuned to achieve the best
        performance and include the desired size of each mini batch, the number
        of epochs to run, and the learning rate eta. If test data is also
        provided, this method will test the performance of the neural network
        after each epoch and print out the number of inputs correctly classified
        against the total amount of test data provided. The test data expects
        the desired outputs to be formatted as a single label as opposed to the
        training data, which formats the desired output as a vector where the
        desired label is indicated by a 1.0 in its corresponding entry in the
        vector.
        '''
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            num_tests = len(test_data)

        for i in range(num_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size]
                            for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_connections(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(i, self.test(test_data),
                                                    num_tests))
            else:
                print("Epoch {} completed.".format(i))

    def update_connections(self, mini_batch, eta):
        '''
        Updates the neural network's weights and biases by using backpropagation
        to implement gradient descent for a single provided mini batch. Instead
        of running backpropagation for each training sample in the mini batch,
        the training samples are transformed into separate matrices, where each
        column in each respective matrix is a single training example in the
        mini batch, and the backpropagation algorithm is run on the resulting
        matrices. The calculated gradients are then used to update the weights
        and biases using the size of the mini batch and the desired learning
        rate.
        '''
        m = len(mini_batch)

        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_w, nabla_b = self.backpropagate(x, y)
        self.weights = [w - (eta / m) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        '''
        Implements the backpropagation algorithm to calculate the gradient of
        the cost function with respect to the weights and biases. The activation
        and z vectors are stored in separate lists during the forward pass of
        the algorithm and are used in the backward pass to calculate the
        appropriate partial derivatives. The variable names are based on their
        respective partial derivatives, where dC_da is the change in the cost
        function relative to a change in the activation, da_dz is the change in
        the activation relative to a change in z, and so on. This is to aid in a
        better understanding of the connection between the backpropagation
        algorithm and the underlying calculus which motivates it. As an extra
        note, the commented out code does not work completely (it requires an if
        statement to avoid IndexErrors) but is included as it more closely
        follows the notation for each layer used in the derivation of the
        algorithm and may aid in understanding what is going on.
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # forward pass
        a = x
        a_ls = [x]
        z_ls = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_ls.append(z)
            a = sigmoid(z)
            a_ls.append(a)

        # backward pass
        dC_da = a_ls[-1] - y # derivative of the cost function
        # multiplying 1 here is included to be precise and show that dz_db is 1
        dC_db = 1 * sigmoid(z_ls[-1], deriv=True) * dC_da
        nabla_w[-1] = np.dot(dC_db, a_ls[-2].transpose())
        nabla_b[-1] = dC_db.sum(axis=1) \
                           .reshape((len(dC_db), 1)) # reshape to n x 1 matrix

        # for l in range(self.num_layers - 2, 0, -1):
        #     da_dz = sigmoid(z_ls[l-1], deriv=True)
        #     dC_db = 1 * np.dot(self.weights[l].transpose(), dC_db) * da_dz
        #     nabla_w[l-1] = np.dot(dC_db, a_ls[l-2].transpose())
        #     nabla_b[l-1] = dC_db.sum(axis=1).reshape((len(dC_db), 1))

        for l in range(2, self.num_layers):
            da_dz = sigmoid(z_ls[-l], deriv=True)
            dC_db = 1 * np.dot(self.weights[-l+1].transpose(), dC_db) * da_dz
            nabla_w[-l] = np.dot(dC_db, a_ls[-l-1].transpose())
            nabla_b[-l] = dC_db.sum(axis=1) \
                               .reshape((len(dC_db), 1))

        return (nabla_w, nabla_b)

    def prediction(self, x):
        '''
        Returns the output vector a calculated by the neural network for the given
        test sample x.
        '''
        a = x
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def test(self, test_data):
        '''
        Returns the number of test samples correctly classified by the neural
        network. The neural network's answer for any particular training sample
        is the index of the neuron in the output layer with the highest
        activation.
        '''
        results = [(np.argmax(self.prediction(x)), y)
                   for x, y in test_data]
        return sum(int(x == y) for x, y in results)

def sigmoid(z, deriv=False):
    '''
    Calculates the output of the sigmoid function or the derivative of the
    sigmoid function for a given vector z.
    '''
    if not deriv:
        return 1 / (1 + np.exp(-z))
    else:
        return (sigmoid(z)) * (1 - sigmoid(z))
