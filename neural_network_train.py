from pathlib import Path
import numpy as np
import pickle

#hyperparameters
DATA_DIR = Path(__file__).resolve().parent / "MNIST"
L = 3 # number of layers, not counting the input layer
image_size = 28 # height and width of image
n = [image_size ** 2, 128, 64, 10] # width of the layers, starting from the input layer
num_train_images = 60000
num_test_images = 10000
alpha = 0.005 # learning rate

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
    return images/255, labels, one_hot_labels


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

def accuracy(empirical, true):
    predictions = np.argmax(empirical, axis = 0)
    num_correct = np.sum(predictions == true)
    percent_correct = num_correct / len(predictions)
    return percent_correct
    
def cross_entropy_loss(empirical, true):
    shifted = empirical - np.max(empirical, axis=0, keepdims=True)
    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=0, keepdims=True)
    )
    return -np.sum(true * log_probs) / true.shape[1] 

def backpropagate(A_array, Z_array, weights, Y):
    """Calculate gradients for summed cross-entropy loss."""
    dweights = [None] * len(weights)
    dbiases = [None] * len(weights)

    # Gradient with respect to the final layer's pre-activation.
    dZ = (A_array[-1] - Y) / Y.shape[1]

    # Work backward from the output layer.
    for k in reversed(range(len(weights))):
        dweights[k] = dZ @ A_array[k].T
        dbiases[k] = np.sum(dZ, axis=1, keepdims=True)

        if k > 0:
            # Propagate through this layer's matrix multiplication.
            dA_previous = weights[k].T @ dZ

            # Propagate through the previous hidden layer's ReLU.
            dZ = dA_previous * (Z_array[k - 1] > 0)

    return dweights, dbiases

def run_epoch(A_previous, costs, accuracies, weights, biases):
    A_array = [A_previous]
    Z_array = []
    for k in range(L):
        Z = weights[k]@A_previous + biases[k]
        if k == L - 1:
            A_previous = activation_output(Z)
        else:
            A_previous = activation_hidden(Z)
        A_array.append(A_previous)
        Z_array.append(Z)
    current_accuracy = accuracy(A_previous, y_train)
    accuracies.append(current_accuracy)
    current_loss = cross_entropy_loss(Z_array[-1], y_train_one_hot)
    costs.append(current_loss)

    dweights, dbiases = backpropagate(
        A_array, Z_array, weights, y_train_one_hot
    )

    for k in range(len(weights)):
        weights[k] -= alpha * dweights[k]
        biases[k] -= alpha * dbiases[k]

    return weights, biases

costs = []
accuracies = []
for m in range(1000):
    weights, biases = run_epoch(x_train, costs, accuracies, weights, biases)
    if m % 10 == 0:
        print(f'Epoch {m}: accuracy = {accuracies[-1]}, loss = {costs[-1]}')
    if len(costs) > 1 and abs(costs[-1] - costs[-2]) < 0.01:
        break

# Save training history and the final model parameters.
results_path = Path(__file__).resolve().parent / "training_results_3layer.pkl"
with results_path.open("wb") as file:
    pickle.dump(
        {
            "costs": costs,
            "accuracies": accuracies,
            "weights": weights,
            "biases": biases,
        },
        file,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
print(f"Saved training results to {results_path}")
