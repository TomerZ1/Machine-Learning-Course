from sklearn.datasets import fetch_openml
import numpy.random
import numpy as np
import matplotlib.pyplot as plt


def knn (train_imgs, labels_vec,  query_img, k): #question a
    dists = calculate_distance(train_imgs, query_img)
    prediction = predict(train_imgs, labels_vec, dists, k)

    return prediction

def calculate_distance(train_imgs, query_img):
    dists = np.linalg.norm(train_imgs - query_img, axis=1)
    return dists

def predict(train_imgs, train_labels, distances, k):
    knn_indices = np.argpartition(distances, k)[:k]
    knn_labels = train_labels[knn_indices]

    unique_labels, counts = np.unique(knn_labels, return_counts=True)
    prediction = unique_labels[np.argmax(counts)]
    return prediction

def evaluate_knn(train_imgs, train_labels, test_imgs, test_labels, k):
    train_labels = train_labels.astype(int)  # Ensure labels are integers
    test_labels = test_labels.astype(int)           # Ensure labels are integers

    correct = 0
    for i in range(len(test_imgs)):
        pred = knn(train_imgs, train_labels, test_imgs[i], k)
        if pred == test_labels[i]:
            correct += 1
    accuracy = correct / len(test_imgs) * 100
    return accuracy

def plot_accuracy_vs_k(train_imgs, train_labels, test_imgs, test_labels, max_k=100): #question c
    train_imgs = train_imgs[:1000]
    train_labels = train_labels[:1000].astype(int)

    test_labels = test_labels.astype(int)
    test_imgs = test_imgs.astype(int)

    # Precompute all distances (as a list of distance vectors)
    print("Precomputing distances...")
    dists_matrix = [calculate_distance(train_imgs, test_img) for test_img in test_imgs]  # list of arrays

    ks = range(1, max_k + 1)
    accuracies = []

    for k in ks:
        correct = 0
        for i in range(len(test_imgs)):
            distances = dists_matrix[i]
            prediction = predict(train_imgs, train_labels, distances, k)
            if prediction == test_labels[i]:
                correct += 1
        accuracy = correct / len(test_imgs)
        accuracies.append(accuracy)
        print(f"k={k}, accuracy={accuracy:.4f}")

    plt.plot(ks, accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("k-NN Accuracy vs k (n=1000)")
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_n(data, labels, test_imgs, test_labels): #question d
    ns = range(100, 5001, 100)
    train_imgs_full = data[:5000]
    train_labels_full = labels[:5000].astype(int)
    test_imgs = test_imgs.astype(int)
    test_labels = test_labels.astype(int)

    dists_matrix = [calculate_distance(train_imgs_full, test_img) for test_img in test_imgs]

    accuracies = []

    for n in ns:
        correct = 0
        for i in range(len(test_imgs)):
            distances = dists_matrix[i][:n]  # Use only first n distances
            prediction = predict(train_imgs_full[:n], train_labels_full[:n], distances, k=1)
            if prediction == test_labels[i]:
                correct += 1
        accuracy = correct / len(test_imgs)
        accuracies.append(accuracy)
        print(f"n={n}, accuracy={accuracy:.4f}")

    plt.plot(ns, accuracies)
    plt.xlabel("Training Set Size (n)")
    plt.ylabel("Accuracy")
    plt.title("k-NN Accuracy vs Training Size (k=1)")
    plt.grid(True)
    plt.show()
