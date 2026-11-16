from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# Hyperparameters
DATA_DIR = Path(__file__).resolve().parent / "MNIST"
L = 3  # Number of layers, not counting the input layer.
image_size = 28
n = [image_size ** 2, 128, 64, 10]
alpha = 0.005
batch_size = 300
epoch_count = 100
dropout_rate = 0.1


def load_mnist(images_filename, labels_filename):
    """Return image columns, digit labels, and one-hot labels."""
    images = np.fromfile(DATA_DIR / images_filename, dtype=np.uint8, offset=16)
    images = images.reshape(-1, image_size ** 2).T / 255.0
    labels = np.fromfile(DATA_DIR / labels_filename, dtype=np.uint8, offset=8)
    one_hot_labels = np.zeros((10, labels.size), dtype=np.uint8)
    one_hot_labels[labels, np.arange(labels.size)] = 1
    return images, labels, one_hot_labels


class MLP(nn.Module):
    def __init__(self, L, n, dropout_rate=0.0):
        super().__init__()
        if L < 1 or len(n) != L + 1:
            raise ValueError("Use at least one layer and L + 1 layer widths.")

        self.dropout = nn.Dropout(p=dropout_rate)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(L):
            scale = (2 / n[i]) ** 0.5 if i < L - 1 else (1 / n[i]) ** 0.5
            self.weights.append(nn.Parameter(torch.randn(n[i], n[i + 1]) * scale))
            self.biases.append(nn.Parameter(torch.zeros(1, n[i + 1])))

    def forward(self, X):
        """Return logits, using inverted dropout on hidden activations during training."""
        self.A_array = [X]
        self.Z_array = []
        self.dropout_masks = []
        for i in range(len(self.weights)):
            Z = X @ self.weights[i] + self.biases[i]
            self.Z_array.append(Z)
            if i < len(self.weights) - 1:
                X = torch.relu(Z)
                # Dropout on ones gives the scaled mask needed by manual backprop.
                mask = self.dropout(torch.ones_like(X))
                self.dropout_masks.append(mask)
                X = X * mask
            else:
                X = Z
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
                dA_previous = (dZ @ self.weights[k].T) * self.dropout_masks[k - 1]
                dZ = dA_previous * (self.Z_array[k - 1] > 0)

        for k in range(len(self.weights)):
            self.weights[k] -= alpha * dweights[k]
            self.biases[k] -= alpha * dbiases[k]
        return dweights, dbiases

    @torch.no_grad()
    def fit(self, X, y, epochs=1000, alpha=0.005, batch_size=128):
        """Train on shuffled mini-batches and return sample-weighted epoch metrics."""
        self.train()
        y = y.to(device=X.device, dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        losses = []
        accuracies = []
        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0
            for X_batch, y_batch in loader:
                Y_batch = torch.nn.functional.one_hot(
                    y_batch, num_classes=self.weights[-1].shape[1]
                ).to(dtype=X_batch.dtype)
                output = self(X_batch)
                loss = self.cross_entropy_loss(output, Y_batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; check inputs and learning rate.")
                total_loss += loss.item() * len(X_batch)
                total_correct += (output.argmax(dim=1) == y_batch).sum().item()
                self.backward(Y_batch, alpha)
            losses.append(total_loss / len(X))
            accuracies.append(total_correct / len(X))
            print(f"Epoch {epoch}: loss = {losses[-1]}, accuracy = {accuracies[-1]}")
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
    model = MLP(L, n, dropout_rate=dropout_rate)
    losses, accuracies = model.fit(
        X_train, y_train, epochs=epoch_count, alpha=alpha, batch_size=batch_size
    )

    torch.save(model.state_dict(), Path(__file__).resolve().parent / f"pytorch_model_dropout{dropout_rate}.pth")
    model.eval()
    weights = model.weights
    biases = model.biases
    with torch.no_grad():
        output = model(X_test)
        test_loss = model.cross_entropy_loss(output, y_test_one_hot).item()
        test_accuracy = model.accuracy(output, y_test).item()
    print(f"Test loss = {test_loss}, test accuracy = {test_accuracy}")

    results_path = Path(__file__).resolve().parent / f"training_results_pytorch_dropout{dropout_rate}.pkl"
    with results_path.open("wb") as file:
        pickle.dump(
            {
                "costs": losses,
                "accuracies": accuracies,
                "weights": weights,
                "biases": biases,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved training results to {results_path}")

    plt.plot(losses, label='loss')
    plt.plot(accuracies, label="accuracy")
    plt.legend()
    plt.show()

#model = MLP(L, n, dropout_rate=dropout_rate)
#losses, accuracies = model.fit(X_train, y_train, epochs=100, alpha=alpha, batch_size=batch_size)