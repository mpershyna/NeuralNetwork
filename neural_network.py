from pathlib import Path
import numpy as np

#hyperparameters
DATA_DIR = Path(__file__).resolve().parent / "MNIST"
L = 2 # number of layers, not counting the input layer
image_size = 28 # height and width of image
n = [image_size ** 2, 128, 10] # width of the layers, starting from the input layer
num_train_images = 60000
num_test_images = 10000

def load_mnist(images_filename, labels_filename):
    """Return image columns, digit labels, and one-hot labels (10, image_count)."""
    # Skip the 16-byte image header, then flatten each image into a column.
    images = np.fromfile(DATA_DIR / images_filename, dtype=np.uint8, offset=16)
    images = images.reshape(-1, image_size ** 2).T

    # Skip the 8-byte label header. Each remaining byte is a digit from 0 to 9.
    labels = np.fromfile(DATA_DIR / labels_filename, dtype=np.uint8, offset=8)

    # Each column has a 1 in the row matching its digit, and 0 elsewhere.
    one_hot_labels = np.zeros((10, labels.size), dtype=np.uint8)
    one_hot_labels[labels, np.arange(labels.size)] = 1
    return images, labels, one_hot_labels


x_train, y_train, y_train_one_hot = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
x_test, y_test, y_test_one_hot = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

weights = []
biases = []
for i in range(L):
    current_weights = np.random.randn(n[i+1], n[i])
    current_biases = np.random.randn(n[i+1], 1)
    weights.append(current_weights)
    biases.append(current_biases)

A_previous = x_train

def activation_hidden(Z):
    """ ReLU activation function for hidden layers """
    return np.maximum(0, Z)

def activation_output(z):
    """ softmax activation function for the final output layer """
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cost(empirical, true):
    predictions = np.argmax(empirical, axis = 0)
    num_correct = np.sum(predictions == true)
    percent_correct = num_correct / len(predictions)
    return percent_correct
    

def run_epoch(A_previous, costs, weights, biases):
    for k in range(L):
        Z = weights[k]@A_previous + biases[k]
        if k == L - 1:
            A_previous = activation_hidden(Z)
        else:
            A_previous = activation_output(Z)
    current_cost = cost(A_previous, y_train)
    costs.append(current_cost)


    


