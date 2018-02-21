import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from plot_decision_regions import plot_decision_regions

iris = load_iris()

X, y = iris.data[:,[2,3]], iris.target
sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm = SVC(kernel='linear', C=100)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, svm, np.array(iris.feature_names)[[2,3]], test_idx=range(90, 150), fileName="svm.png")

print( "Accuracy: {}".format(accuracy_score(y_test, predictions)) )
