"""
Creates a scatter plot for a 2D dataset
The format of dataset is a list of the following entries
x <space> y <space> cluster_number
x and y are floating point values

Usage:
	python3 path/to/this/script.py path/to/dataset
"""

from matplotlib import pyplot as plt
import sys

data = {}

for line in open(sys.argv[1], 'r'):
    x, y, n = line.split()
    x = float(x)
    y = float(y)

    if n not in data:  # Create an entry if none exists
        data[n] = [[], []]

    data[n][0].append(x)
    data[n][1].append(y)

for n in data:
    plt.scatter(data[n][0], data[n][1], label="Cluster " + str(n))

plt.legend(loc="upper right")
plt.show()

