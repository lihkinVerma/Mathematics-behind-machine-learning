import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


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


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, [1, 3]]  # we only take the first and the third features.
Y = iris.target

# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-90]]
Y_train = Y[indices[:-90]]
X_test = X[indices[-90:]]
Y_test = Y[indices[-90:]]

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 100  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),  # 
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X_train, Y_train) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.6, hspace=0.6)

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for svm, title, ax in zip(models, titles, sub.flatten()):
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    plot_contours(ax, svm, xx, yy, cmap=plt.cm.brg, alpha=0.8)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.brg, s=20, edgecolors='k', marker='D')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title + ' \n traning error: ' + str(
        round(100 * (1 - svm.score(X_train, Y_train)), 2)) + '%' ' \n test error: ' + str(
        round(100 * (1 - svm.score(X_test, Y_test)), 2)) + '%')

plt.show()
