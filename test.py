import mnist_loader
import neural_network

training_data, validation_data, test_data = mnist_loader.load_data()
network = neural_network.NeuralNetwork([784, 30, 10])
network.train(training_data, 30, 10, 3.0, test_data=test_data)
