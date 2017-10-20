import csv
import sys
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from ex3 import plot_feature_importances
def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])
    # Extract feature names
    feature_names = np.array(X[0])
    # Remove the first row because they are feature names
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names
X, y, feature_names = load_dataset(sys.argv[1])
X, y = shuffle(X, y, random_state=7)
