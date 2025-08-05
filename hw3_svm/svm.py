import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100



# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1,X2],axis=1)
y = np.concatenate([np.ones((n,1)), -np.ones((n,1))], axis=0).reshape([-1])


def question_a_b(q):
    # if q=0 so homogeneous polynomial kernel, if q=1 so inhomogeneous polynomial kernel
    clf_linear = svm.SVC(kernel="linear", C=10)
    clf_linear.fit(X, y)

    clf_poly2 = svm.SVC(kernel="poly", degree=2, coef0=q, C=10)
    clf_poly2.fit(X, y)

    clf_poly3 = svm.SVC(kernel="poly", degree=3, coef0=q, C=10)
    clf_poly3.fit(X, y)

    models = [clf_linear, clf_poly2, clf_poly3]
    titles = ["Linear", "Polynomial degree 2", "Polynomial degree 3"]

    plot_results(models, titles, X, y)

def question_c(g):

    # flip 10% of the negative labels to positive labels
    y_with_noise = y.copy()
    neg_indices = np.where(y_with_noise == -1)[0]
    num_to_flip = int(len(neg_indices) * 0.1)
    flip_indices = np.random.choice(neg_indices, num_to_flip, replace=False)
    y_with_noise[flip_indices] = 1

    clf_poly2 = svm.SVC(kernel="poly", degree=2, coef0=1, C=10)
    clf_poly2.fit(X, y_with_noise)

    clf_rbf = svm.SVC(kernel="rbf", gamma=g, C=10)
    clf_rbf.fit(X, y_with_noise)

    models = [clf_poly2, clf_rbf]
    titles = ["Polynomial degree 2", "RBF kernel"]
    plot_results(models, titles, X, y_with_noise)

    return 0

#question_a_b(1) # Question A
#question_a_b(1) # Question B     
#question_c(10) # Question C  

