import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import ex12

# Load input data
# input_file = 'iris2.txt'
input_file = 'data.txt'
X, y = ex12.load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black',edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None',edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
# params = {'kernel': 'linear'}
params = {'kernel': 'linear','class_weight': 'balanced'}
# params = {'kernel': 'poly', 'degree': 3}
# params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

ex12.plot_classifier(classifier, X_train, y_train, 'Trainingdataset')
plt.show()

y_test_pred = classifier.predict(X_test)
ex12.plot_classifier(classifier, X_test, y_test, 'Testdataset')
plt.show()

target_names = ['Class-' + str(int(i)) for i in set(y)]
print "\n" + "#"*30
print "\nClassifier performance on training dataset\n"
print classification_report(y_train, classifier.predict(X_train),target_names=target_names)
print "#"*30 + "\n"

# Measure distance from the boundary
input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4],
[2.5, 7], [7.6, 2], [5.4, 5.9]])

print "\nDistance from the boundary:"
for i in input_datapoints:
    print i, '-->', classifier.decision_function([i])[0]

# Confidence measure
params = {'kernel': 'rbf', 'probability': True}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

print "\nConfidence measure:"
for i in input_datapoints:
    print i, '-->', classifier.predict_proba([i])[0]
ex12.plot_classifier(classifier, input_datapoints,[0]*len(input_datapoints), 'Input datapoints', 'True')
plt.show()

# Set the parameters by cross-validation
parameter_grid = [{'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                  {'kernel': ['poly'], 'degree': [2, 3]},
                  {'kernel': ['rbf'], 'gamma': [0.01, 0.001],
                   'C': [1, 10, 50, 600]},
                  ]
metrics = ['precision', 'recall_weighted']
for metric in metrics:
    print "\n#### Searching optimal hyperparameters for", metric
    classifier = model_selection.GridSearchCV(SVC(C=1),parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)
    print "\nScores across the parameter grid:"
    for params,avg_score in zip(classifier.cv_results_["params"],classifier.cv_results_["mean_test_score"]):
        print params, '-->', round(avg_score, 3)
    print "\nHighest scoring parameter set:", classifier.best_params_
    y_true, y_pred = y_test, classifier.predict(X_test)
    print "\nFull performance report:\n"
    print classification_report(y_true, y_pred)