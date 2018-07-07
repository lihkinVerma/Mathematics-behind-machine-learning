import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets,linear_model,discriminant_analysis,tree, dummy, neighbors
# from sklearn.neighbors import KNeighborsClassifier
import graphviz
import collections
import pydotplus

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

iris = datasets.load_iris()
X = iris.data[:, [1,3]]  # we only take the first and the third features.
Y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-30]]
Y_train = Y[indices[:-30]]
X_test  = X[indices[-30:]]
Y_test  = Y[indices[-30:]]

C = 100  # SVM regularization parameter
models = (linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1),
		discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001),
		tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.10, min_impurity_split=None, class_weight=None, presort=False),
		svm.SVC(kernel='linear', C=C),
      	svm.LinearSVC(C=C),
      	svm.SVC(kernel='rbf', gamma=0.7, C=C),
      	svm.SVC(kernel='poly', degree=3, C=C),
      	neighbors.KNeighborsClassifier(3),
      	dummy.DummyClassifier(strategy='uniform'))
models = (clf.fit(X_train, Y_train) for clf in models)

titles = ('Logistic Regression',
		'LDA',
		'Decision Tree',
		'SVC with linear kernel', # one vs one
      	'LinearSVC (linear kernel)', #one vs all
      	'SVC with RBF kernel',
      	'SVC with polynomial (degree 3) kernel',
      	'k-NN',
      	'Random Classifier')

fig, sub = plt.subplots(3, 3)
plt.subplots_adjust(wspace=0.6, hspace=0.9)

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for svm, title, ax in zip(models, titles, sub.flatten()):
    
	Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
	plot_contours(ax, svm, xx, yy, cmap=plt.cm.brg, alpha=0.8)
	#Z = Z.reshape(xx.shape)
	#plt.figure(1, figsize=(4, 3))
	#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	#ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.brg, s=20, edgecolors='k')
	ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.brg, s=20, edgecolors='k',marker = 'D')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xlabel('Sepal length')
	ax.set_ylabel('Sepal width')
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_title(title + ' \n traning error: ' + str(round(100*(1-svm.score(X_train, Y_train)),2))+'%' ' \n test error: ' + str(round(100*(1-svm.score(X_test, Y_test)),2))+'%')

plt.show()