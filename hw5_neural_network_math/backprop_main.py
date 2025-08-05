import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)


# Training configuration
epochs = 30
batch_size = 100
learning_rate = 0.1

# Network configuration
#layer_dims = [784, 40, 10]
#net = Network(layer_dims)
#net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)


def question_b():
    n_train = 10000
    n_test = 5000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
    learning_rates = [0.001, 0.01, 0.1, 1, 10]
    epochs = 30
    batch_size = 10
    all_train_accuracies = {}
    all_train_losses = {}
    all_test_accuracies = {}
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        net = Network([784, 40, 10])
        _, train_losses, _, train_accs, test_accs = net.train(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            x_test=x_test,
            y_test=y_test
        )
        all_train_accuracies[lr] = train_accs
        all_train_losses[lr] = train_losses
        all_test_accuracies[lr] = test_accs

    epochs_range = range(epochs)

    # Plot training accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    for lr in learning_rates:
        plt.plot(epochs_range, all_train_accuracies[lr], label=f"lr={lr}")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot training loss
    plt.subplot(1, 3, 2)
    for lr in learning_rates:
        plt.plot(epochs_range, all_train_losses[lr], label=f"lr={lr}")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 3, 3)
    for lr in learning_rates:
        plt.plot(epochs_range, all_test_accuracies[lr], label=f"lr={lr}")
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def question_c():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    net = Network([784, 40, 10])
    _, _, _, _, test_accs = net.train(
        x_train, y_train,
        epochs=30,
        batch_size=100,
        learning_rate=0.1,
        x_test=x_test,
        y_test=y_test
    )

    print(f"Final test accuracy: {test_accs[-1]}")

def question_d():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
    x_train = x_train.T
    x_test = x_test.T   
    W = np.random.randn(10, 784) * 0.01
    epochs = 30
    learning_rate = 0.1
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Compute logits
        train_logits = x_train @ W.T
        test_logits = x_test @ W.T

        # Softmax probabilities
        train_probs = np.exp(train_logits)
        train_probs /= train_probs.sum(axis=1, keepdims=True)

        test_probs = np.exp(test_logits)
        test_probs /= test_probs.sum(axis=1, keepdims=True)

        # Predictions
        train_preds = np.argmax(train_probs, axis=1)
        test_preds = np.argmax(test_probs, axis=1)

        # Accuracy
        train_acc = np.mean(train_preds == y_train)
        test_acc = np.mean(test_preds == y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # One-hot encode y_train
        y_one_hot = np.zeros_like(train_probs)
        y_one_hot[np.arange(len(y_train)), y_train] = 1

        # Compute gradient of loss w.r.t W
        grad_logits = (train_probs - y_one_hot) / len(y_train)  # (50000, 10)
        grad_W = grad_logits.T @ x_train  # (10, 784)

        # Gradient descent step
        W -= learning_rate * grad_W

    # Plot each row of W as an image
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(W[i].reshape((28, 28)), interpolation='nearest', cmap='gray')
        plt.title(f"Class {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Plot accuracy curves
    plt.plot(range(epochs), train_accuracies, label="Train Accuracy")
    plt.plot(range(epochs), test_accuracies, label="Test Accuracy")
    plt.title("Accuracy over Epochs (Linear Classifier)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #question_b()
    #question_c()
    question_d()