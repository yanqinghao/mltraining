import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_data_to_timeseries(input_file, column, verbose=False):
    # Load the input file
    data = np.loadtxt(input_file, delimiter=',')
    # Extract the start and end dates
    start_date = str(int(data[0, 0])) + '-' + str(int(data[0, 1]))
    end_date = str(int(data[-1, 0] + 1)) + '-' + str(int(data[-1, 1] % 12 + 1))
    if verbose:
        print "\nStart date =", start_date
        print "End date =", end_date
    # Create a date sequence with monthly intervals
    dates = pd.date_range(start_date, end_date, freq='M')
    # Convert the data into time series data
    data_timeseries = pd.Series(data[:, column], index=dates)
    if verbose:
        print "\nTime series data:\n", data_timeseries[:10]
    return data_timeseries

if __name__=='__main__':
    # Input file containing data
    input_file = 'data_timeseries.txt'
    # Load input data
    column_num = 2
    data_timeseries = convert_data_to_timeseries(input_file,column_num,True)
    # Plot the time series data
    data_timeseries.plot()
    plt.title('Input data')
    plt.show()

    # Plot within a certain year range
    start = '2008'
    end = '2015'
    plt.figure()
    data_timeseries[start:end].plot()
    plt.title('Data from ' + start + ' to ' + end)
    # Plot within a certain range of dates
    start = '2007-2'
    end = '2007-11'
    plt.figure()
    data_timeseries[start:end].plot()
    plt.title('Data from ' + start + ' to ' + end)
    plt.show()

    # Load data
    data1 = convert_data_to_timeseries(input_file, 2)
    data2 = convert_data_to_timeseries(input_file, 3)
    dataframe = pd.DataFrame({'first': data1, 'second': data2})
    # Plot data
    dataframe['1952':'1955'].plot()
    plt.title('Data overlapped on top of each other')
    # Plot the difference
    plt.figure()
    difference = dataframe['1952':'1955']['first'] - dataframe['1952':'1955']['second']
    difference.plot()
    plt.title('Difference (first - second)')
    # When 'first' is greater than a certain threshold
    # and 'second' is smaller than a certain threshold
    dataframe[(dataframe['first'] > 60) & (dataframe['second'] < 20)].plot()
    plt.title('first > 60 and second < 20')
    plt.show()

    # Print max and min
    print '\nMaximum:\n', dataframe.max()
    print '\nMinimum:\n', dataframe.min()
    # Print mean
    print '\nMean:\n', dataframe.mean()
    print '\nMean row-wise:\n', dataframe.mean(1)[:10]
    # Plot rolling mean
    dataframe.rolling(window=24).mean().plot()
    # Print correlation coefficients
    print '\nCorrelation coefficients:\n', dataframe.corr()
    # Plot rolling correlation
    plt.figure()
    dataframe['first'].rolling(window=60).corr(dataframe['second']).plot()
    plt.show()