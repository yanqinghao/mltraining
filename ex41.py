# import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.finance import quotes_historical_yahoo_ochl
from hmmlearn.hmm import GaussianHMM

params = {'tickers': 'INTC', 'begin': '1994-04-05', 'end': '2015-07-03'}
r = requests.get('https://quantprice.herokuapp.com/api/v1.1/scoop/period', params=params)
data = r.json()
quotes = np.array(data["datatable"]["data"])[:,-6:]

# Get quotes from Yahoo finance
# quotes = quotes_historical_yahoo_ochl("INTC",datetime.date(1994, 4, 5), datetime.date(2015, 7, 3))

# Extract the required values
dates = quotes[:,0]
closing_values = quotes[:,4]
volume_of_shares = quotes[1:,5]
# Take diff of closing values and computing rate of change
diff_percentage = 100.0 * np.diff(closing_values) / closing_values[:-1]
dates = dates[1:]
# Stack the percentage diff and volume values column-wise for training
X = np.column_stack([diff_percentage, volume_of_shares])
X = [x for x in X if x[0]!=None and x[1]!=None]
# Create and train Gaussian HMM
print "\nTraining HMM...."
model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
model.fit(X)
# Generate data using model
num_samples = 500
a, samples = model.sample(num_samples)
plt.plot(np.arange(num_samples), samples, c='black')
plt.show()