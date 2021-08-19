import pickle
import numpy as np
from tensorflow.keras.datasets import mnist
from neural_net import NeuralNet
from functions import accuracy_score


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, -1)/255.0
X_test = X_test.reshape(10000, -1)/255.0

with open('sample_weight.pkl', 'rb') as f:
    sample = pickle.load(f)


mnist_net = NeuralNet()
mnist_net.add(parameters=sample)
mnist_net.fit(X_test, ['sigmoid', 'sigmoid', 'softmax'])
y_pred = mnist_net.predict()

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)