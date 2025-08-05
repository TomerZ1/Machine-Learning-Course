#################################
# Your name: Tomer Zalberg 
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n, d = data.shape
    w = np.zeros(d)
    for t in range(1, T+1):
        i = np.random.randint(n)
        x_i = data[i]
        y_i = labels[i]
        eta_t = eta_0 / t   
        margin = y_i * np.dot(w, x_i)
        if margin < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w
    return w



#################################

# Place for additional code

#################################

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

def q_1_a():
    etas = [10 ** i for i in range(-5, 6)]
    accuracies = []
    for eta_0 in etas:
        sum_acc = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C=1, eta_0=eta_0, T=1000)
            acc = calc_accuracy(validation_data, validation_labels, w)
            sum_acc += acc
        avg_acc = sum_acc / 10
        accuracies.append(avg_acc)
    
    best_eta = etas[np.argmax(accuracies)]
    plt.semilogx(etas, accuracies, marker='o')
    plt.xlabel("η₀ (log scale)")
    plt.ylabel("Average validation accuracy")
    plt.title("SGD Hinge: Validation Accuracy vs. η₀")
    plt.grid(True)
    plt.show()

    return best_eta

def q_1_b():
    cs = [10 ** i for i in range(-5, 6)]
    accuracies = []
    eta_0 = 1 # from previous question its 10^0=1 based on the plot
    for c in cs:
        sum_acc = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta_0, T=1000)
            acc = calc_accuracy(validation_data, validation_labels, w)
            sum_acc += acc
        avg_acc = sum_acc / 10
        accuracies.append(avg_acc)

    best_c = cs[np.argmax(accuracies)]
    plt.semilogx(cs, accuracies, marker='o')
    plt.xlabel("C (log scale)")
    plt.ylabel("Average validation accuracy")
    plt.title("SGD Hinge: Validation Accuracy vs. C")
    plt.grid(True)
    plt.show()

    return best_c

def q_1_c():
    eta_0 = 1 # from previous question its 10^0=1 based on the plot
    c = 0.0001 # from previous question based on the plot
    T = 20000
    w = SGD_hinge(train_data, train_labels, c, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

def q_1_d():
    w = SGD_hinge(train_data, train_labels, C=0.0001, eta_0=1, T=20000)
    acc = calc_accuracy(test_data, test_labels, w)
    print(f"Test accuracy: {acc:.4f}")

def calc_accuracy(X, y, w):
    predictions = np.sign(X @ w)
    return np.mean(predictions == y)

#print(q_1_a())
#print(q_1_b())
#q_1_c()
#q_1_d()