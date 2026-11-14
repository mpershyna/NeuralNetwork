from pathlib import Path
import torch
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler

# Hyperparameters
DATA_DIR = Path(__file__).resolve().parent / "MNIST"
L = 3  # Number of layers, not counting the input layer.
image_size = 28
n = [image_size ** 2, 128, 64, 10]
alpha = 0.005


def load_mnist(images_filename, labels_filename):
    """Return image columns, digit labels, and one-hot labels."""
    images = np.fromfile(DATA_DIR / images_filename, dtype=np.uint8, offset=16)
    images = images.reshape(-1, image_size ** 2).T / 255.0
    labels = np.fromfile(DATA_DIR / labels_filename, dtype=np.uint8, offset=8)
    one_hot_labels = np.zeros((10, labels.size), dtype=np.uint8)
    one_hot_labels[labels, np.arange(labels.size)] = 1
    return images, labels, one_hot_labels


class MLP(nn.Module):
    def __init__(self, L, n):
        super().__init__()
        if L < 1 or len(n) != L + 1:
            raise ValueError("Use at least one layer and L + 1 layer widths.")

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(L):
            scale = (2 / n[i]) ** 0.5 if i < L - 1 else (1 / n[i]) ** 0.5
            self.weights.append(nn.Parameter(torch.randn(n[i], n[i + 1]) * scale))
            self.biases.append(nn.Parameter(torch.zeros(1, n[i + 1])))

    def forward(self, X):
        """Return logits; each input row is one image."""
        self.A_array = [X]
        self.Z_array = []
        for i in range(len(self.weights)):
            Z = X @ self.weights[i] + self.biases[i]
            self.Z_array.append(Z)
            X = torch.relu(Z) if i < len(self.weights) - 1 else Z
            self.A_array.append(X)
        return X

    def accuracy(self, empirical, true):
        """Return the fraction correct; true is a vector of digit labels."""
        predictions = torch.argmax(empirical, dim=1)
        return torch.sum(predictions == true) / len(predictions)

    def cross_entropy_loss(self, empirical, true):
        """Return mean cross-entropy from logits and row-wise one-hot labels."""
        shifted = empirical - torch.amax(empirical, dim=1, keepdim=True)
        log_probs = shifted - torch.log(
            torch.sum(torch.exp(shifted), dim=1, keepdim=True)
        )
        return -torch.sum(true * log_probs) / true.shape[0]

    @torch.no_grad()
    def backward(self, Y, alpha=0.005):
        """Calculate all gradients, then update parameters; Y is one-hot."""
        dweights = [None] * len(self.weights)
        dbiases = [None] * len(self.weights)
        dZ = (torch.softmax(self.A_array[-1], dim=1) - Y) / Y.shape[0]

        for k in reversed(range(len(self.weights))):
            dweights[k] = self.A_array[k].T @ dZ
            dbiases[k] = torch.sum(dZ, dim=0, keepdim=True)
            if k > 0:
                dZ = (dZ @ self.weights[k].T) * (self.Z_array[k - 1] > 0)

        for k in range(len(self.weights)):
            self.weights[k] -= alpha * dweights[k]
            self.biases[k] -= alpha * dbiases[k]
        return dweights, dbiases

    @torch.no_grad()
    def fit(self, X, y, epochs=1000, alpha=0.005):
        """Train on the supplied images and digit labels using manual gradients."""
        self.train()
        y = y.to(device=X.device, dtype=torch.long)
        Y = torch.nn.functional.one_hot(
            y, num_classes=self.weights[-1].shape[1]
        ).to(dtype=X.dtype)
        losses = []
        accuracies = []
        for epoch in range(epochs):
            output = self(X)
            loss = self.cross_entropy_loss(output, Y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check inputs and learning rate.")
            acc = self.accuracy(output, y).item()
            losses.append(loss.item())
            accuracies.append(acc)
            self.backward(Y, alpha)
            print(f"Epoch {epoch}: loss = {losses[-1]}, accuracy = {acc}")
        return losses, accuracies


if __name__ == "__main__":
    x_train, y_train, _ = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    x_test, y_test, y_test_one_hot = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(x_train.T), dtype=torch.float32)
    X_test = torch.tensor(scaler.transform(x_test.T), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    y_test_one_hot = torch.tensor(y_test_one_hot.T, dtype=torch.float32)

    torch.manual_seed(42)
    model = MLP(L, n)
    losses, accuracies = model.fit(X_train, y_train, epochs=10, alpha=alpha)

    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = model.cross_entropy_loss(output, y_test_one_hot).item()
        test_accuracy = model.accuracy(output, y_test).item()
    print(f"Test loss = {test_loss}, test accuracy = {test_accuracy}")

model = MLP(L, n)
losses, accuracies = model.fit(X_train, y_train, epochs=10, alpha=alpha)