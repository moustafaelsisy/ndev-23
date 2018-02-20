from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from plot_decision_regions import plot_decision_regions

iris = load_iris()
X,y = iris.data[:,[0,2]], iris.target	# two features only
# X,y = iris.data[:,:], iris.target	# all features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

rf = RandomForestClassifier(n_estimators=40, criterion="gini")
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, rf, np.array(iris.feature_names)[[2,3]], test_idx=range(90,150), fileName="rf.png")

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}".format(accuracy))