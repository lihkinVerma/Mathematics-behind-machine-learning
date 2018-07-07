import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, [1, 3]]  # we only take the first and the third features.
Y = iris.target

# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-30]]
Y_train = Y[indices[:-30]]
X_test  = X[indices[-30:]]
Y_test  = Y[indices[-30:]]

logreg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
# check each parameter (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# we create an instance of Neighbours Classifier and fit the data.
model = logreg.fit(X_train, Y_train)
print('LogisticRegression score: %f' % model.score(X_test, Y_test))

# Plot the decision boundary. For that, we will assign a color to each
h = .02  # step size in the mesh

# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k', cmap=plt.cm.Paired) # black border for training data point
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='r', cmap=plt.cm.Paired) # red border for test data point
plt.xlabel('Petal length')
plt.ylabel('Sepal length')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
