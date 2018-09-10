import _pickle
import gzip
import numpy as np

def load_data():
    '''
    Loads data from the MNIST database and transforms it into a convenient
    format for use by the neural network. In particular, the image data is
    transformed from a 28 x 28 matrix of pixels into a 784 x 1 matrix for the
    input layer of the network. For use in training, the provided labels in the
    training data are transformed into a 10 x 1 vector containing a 1.0 in the
    specified digit's position and 0.0's otherwise.
    '''
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding='latin1')
    f.close()

    training_x = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_y = [vectorize_labels(y) for y in training_data[1]]
    training_data = zip(training_x, training_y)

    validation_x = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_x, validation_data[1])

    test_x = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_x, test_data[1])

    return (training_data, validation_data, test_data)

def vectorize_labels(label):
    v = np.zeros((10, 1))
    v[label] = 1.0
    return v
