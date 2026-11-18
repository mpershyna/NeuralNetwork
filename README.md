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

## Tuning process
- Initial simple MLP (Numpy-based) achieved 66% test accuracy. This used 1 hidden layer (128 width) and 1 output layer, ReLU activation for the hidden layer, softmax activation for the output layer, cross-entropy loss, and gradient descent with analytically computed gradients.
- An additional hidden layer of width 64 improved the test accuracy to approximately 80% (Numpy-based model)
- The same model implemented in Pytorch (with the addition of re-scaling the data to have mean 0 and variance 1) led to an improved test accuracy of 88% and a much faster training
- Increasing the width of the first hidden layer to 500 did not significantly improve the train or test accuracy but did lead to longer training, so I kept the original 128 width for the first hidden layer.
- Introducing batches of size 100 into the training improved training accuracy to over 99% but test accuracy lagged at approximately 96.5%, suggesting some overfitting. Batch sizes of 300 and 600 led to slightly lower train and test accuracy but still had training accuracy higher by 1-2%. 
- Introducing dropout plus a batch size of 50 led to a test accuracy of 97.4%
- Optimal parameters were found to be a batch size of 100 and step size of 0.1 with the best performance on the validation set. Test accuracy with these parameters is 97.6% (training accuracy 99.6%, validation accuracy 97.5%).
