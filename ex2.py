import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import cPickle as pickle
from sklearn.preprocessing import PolynomialFeatures

filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(' ')]
        X.append(xt)
        y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])
# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
linear_regressor = linear_model.LinearRegression()
# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

print "Mean absolute error =", round(sm.mean_absolute_error(y_test,y_test_pred), 2)
print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)
print "Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)
print "R2 score =", round(sm.r2_score(y_test, y_test_pred), 2)

output_model_file = 'saved_model.pkl'
with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f:
    model_linregr = pickle.load(f)
y_test_pred_new = model_linregr.predict(X_test)
print "\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2)

ridge_regressor = linear_model.Ridge(alpha=0.1, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2)
print "Mean squared error =", round(sm.mean_squared_error(y_test,y_test_pred_ridge), 2)
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2)
print "Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2)
print "R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge),2)

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[5.1]]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print "\nLinear regression:", linear_regressor.predict(datapoint)[0]
print "\nPolynomial regression:", poly_linear_model.predict(poly_datapoint)[0]