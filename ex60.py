import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Define the number of values
n = 250
# Create a lambda function to generate the random values in the given range
f = lambda minval, maxval, n: minval + (maxval - minval) * np.random.rand(n)
# Generate the values
x_vals = f(15, 41, n)
y_vals = f(-10, 70, n)
z_vals = f(-52, -37, n)
# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c='k', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# Define the number of values
num_vals = 40
# Generate random values
x = np.random.rand(num_vals)
y = np.random.rand(num_vals)
# Define area for each bubble
# Max radius is set to a specified value
max_radius = 25
area = np.pi * (max_radius * np.random.rand(num_vals)) ** 2
# Generate colors
colors = np.random.rand(num_vals)
# Plot the points
plt.scatter(x, y, s=area, c=colors, alpha=1.0)
plt.show()

# Labels and corresponding values in counter clockwise direction
data = {'Apple': 26,
'Mango': 17,
'Pineapple': 21,
'Banana': 29,
'Strawberry': 11}
# List of corresponding colors
colors = ['orange', 'lightgreen', 'lightblue', 'gold', 'cyan']
# Needed if we want to highlight a section
explode = (0.1, 0, 0, 0, 0)
# Plot the pie chart
plt.figure()
plt.pie(data.values(), explode=explode, labels=data.keys(),colors=colors, autopct='%1.1f%%', shadow=False,startangle=90)
# Aspect ratio of the pie chart, 'equal' indicates tht we
# want it to be a circle
plt.axis('equal')
plt.show()

def tracker(cur_num):
    # Get the current index
    cur_index = cur_num % num_points
    # Set the color of the datapoints
    datapoints['color'][:, 3] = 1.0
    # Update the size of the circles
    datapoints['size'] += datapoints['growth']
    # Update the position of the oldest datapoint
    datapoints['position'][cur_index] = np.random.uniform(0, 1, 2)
    datapoints['size'][cur_index] = 7
    datapoints['color'][cur_index] = (0, 0, 0, 1)
    datapoints['growth'][cur_index] = np.random.uniform(40, 150)
    # Update the parameters of the scatter plot
    scatter_plot.set_edgecolors(datapoints['color'])
    scatter_plot.set_sizes(datapoints['size'])
    scatter_plot.set_offsets(datapoints['position'])

if __name__=='__main__':
    # Create a figure
    fig1 = plt.figure(figsize=(9, 7), facecolor=(0,0.9,0.9))
    ax = fig1.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])
    # Create and initialize the datapoints in random positions
    # and with random growth rates.
    num_points = 20
    datapoints = np.zeros(num_points, dtype=[('position', float,2),('size', float, 1),
                                             ('growth', float, 1), ('color',float, 4)])
    datapoints['position'] = np.random.uniform(0, 1, (num_points,2))
    datapoints['growth'] = np.random.uniform(40, 150, num_points)
    # Construct the scatter plot that will be updated every frame
    scatter_plot = ax.scatter(datapoints['position'][:, 0],datapoints['position'][:, 1],
                              s=datapoints['size'], lw=0.7,edgecolors=datapoints['color'],facecolors='none')
    # Start the animation using the 'tracker' function
    animation = FuncAnimation(fig1, tracker, interval=10)
    plt.show()

