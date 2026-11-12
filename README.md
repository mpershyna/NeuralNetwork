# NeuralNetwork

A neural network built from scratch with NumPy to classify handwritten MNIST digits. Work in progress.

## Setup

From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## MNIST data

Place the uncompressed MNIST files in a local `MNIST` folder:

```text
MNIST/
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
```

The dataset and virtual environment are excluded from Git.

## Code

- `neural_network.py`: flattened MNIST images, one-hot labels, and the neural network implementation in progress.
- `mnist_loader.py`: the original MNIST loading and image display example.

Run `python neural_network.py` to load the data and initialize the network. Training is still being implemented.
