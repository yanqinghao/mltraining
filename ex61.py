import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import matplotlib.cbook as cbook
from matplotlib.ticker import Formatter
import matplotlib.animation as animation

# Define a class for formatting
class DataFormatter(Formatter):
    def __init__(self, dates, date_format='%Y-%m-%d'):
        self.dates = dates
        self.date_format = date_format

    # Extact the value at time t at position 'position'
    def __call__(self, t, position=0):
        index = int(round(t))
        if index >= len(self.dates) or index < 0:
            return ''
        return self.dates[index].strftime(self.date_format)

# Generate the signal
def generate_data(length=2500, t=0, step_size=0.05):
    for count in range(length):
        t += step_size
        signal = np.sin(2*np.pi*t)
        damper = np.exp(-t / 8.0)
        yield t, signal * damper

# Initializer function
def initializer():
    peak_val = 1.0
    buffer_val = 0.1
    ax.set_ylim(-peak_val * (1 + buffer_val), peak_val * (1 + buffer_val))
    ax.set_xlim(0, 10)
    del x_vals[:]
    del y_vals[:]
    line.set_data(x_vals, y_vals)
    return line

def draw(data):
    # update the data
    t, signal = data
    x_vals.append(t)
    y_vals.append(signal)
    x_min, x_max = ax.get_xlim()
    if t >= x_max:
        ax.set_xlim(x_min, 2 * x_max)
        ax.figure.canvas.draw()
    line.set_data(x_vals, y_vals)
    return line

if __name__=='__main__':
    # CSV file containing the stock quotes
    input_file = cbook.get_sample_data('aapl.csv',asfileobj=False)
    # Load csv file into numpy record array
    data = csv2rec(input_file)
    # Take a subset for plotting
    data = data[-70:]
    a = data.date
    # Create the date formatter object
    formatter = DataFormatter(data.date)
    # X axis
    x_vals = np.arange(len(data))
    # Y axis values are the closing stock quotes
    y_vals = data.close
    # Plot data
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(x_vals, y_vals, 'o-')
    fig.autofmt_xdate()
    plt.show()

    # Input data
    apples = [30, 25, 22, 36, 21, 29]
    oranges = [24, 33, 19, 27, 35, 20]
    # Number of groups
    num_groups = len(apples)
    # Create the figure
    fig, ax = plt.subplots()
    # Define the X axis
    indices = np.arange(num_groups)
    # Width and opacity of histogram bars
    bar_width = 0.4
    opacity = 0.6
    # Plot the values
    hist_apples = plt.bar(indices, apples, bar_width,alpha=opacity, color='g', label='Apples')
    hist_oranges = plt.bar(indices + bar_width, oranges, bar_width,alpha=opacity, color='b', label='Oranges')
    plt.xlabel('Month')
    plt.ylabel('Production quantity')
    plt.title('Comparing apples and oranges')
    plt.xticks(indices + bar_width, ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'))
    plt.ylim([0, 45])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Define the two groups
    group1 = ['France', 'Italy', 'Spain', 'Portugal', 'Germany']
    group2 = ['Japan', 'China', 'Brazil', 'Russia', 'Australia']
    # Generate some random values
    data = np.random.rand(5, 5)
    # Create a figure
    fig, ax = plt.subplots()
    # Create the heat map
    heatmap = ax.pcolor(data, cmap=plt.cm.gray)
    # Add major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    # Make it look like a table
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # Add tick labels
    ax.set_xticklabels(group2, minor=False)
    ax.set_yticklabels(group1, minor=False)
    plt.show()

    # Create the figure
    fig, ax = plt.subplots()
    ax.grid()
    # Extract the line
    line, = ax.plot([], [], lw=1.5)
    # Create the variables
    x_vals, y_vals = [], []
    # Define the animator object
    animator = animation.FuncAnimation(fig, draw, generate_data,blit=False,
                                       interval=10, repeat=False, init_func = initializer)
    plt.show()