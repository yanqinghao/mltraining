import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.svm import SVC
import ex12

# Load input data
input_file = 'iris2.txt'
X, y = ex12.load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black',edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None',edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
params = {'kernel': 'linear'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

ex12.plot_classifier(classifier, X_train, y_train, 'Trainingdataset')
plt.show()