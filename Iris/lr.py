import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from plot_decision_regions import plot_decision_regions

iris = load_iris()

X, y = iris.data[:,[0,2]], iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

lr = LogisticRegression(C=1000)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, lr, np.array(iris.feature_names)[[2,3]], test_idx=range(90, 150), fileName="lr.png")

print( "Accuracy: {}".format(accuracy_score(y_test, predictions)) )
