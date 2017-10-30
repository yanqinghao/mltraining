import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import ex16

# Load data from input file
X = ex16.load_data('data_multivar.txt')

# Estimating the bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
# Compute clustering with MeanShift
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
