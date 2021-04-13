"""
Creates blob datasets of required number of datapoints and centers
Usage: python3 path/to/this/file.py <n> <k> [<d>]
where n is the number of datapoints, k is number of centers and d (default d=2) is number of dimentions
"""


from sklearn.datasets import make_blobs
import sys

if(len(sys.argv) < 3):
	print("Usage: python3 path/to/this/file.py <n> <k> [<d>]")
	print("where n is the number of datapoints, k is number of centers and d (default d=2) is number of dimentions")

n = int(sys.argv[1])
k = int(sys.argv[2])
d = 2
if(len(sys.argv) == 4):
	d = int(sys.argv[3])

X, Y = make_blobs(n_samples=n, centers=k, n_features=d)

F = open("BigData_"+str(n)+".txt", "w")
for n in range(X.shape[0]):
	s = ""
	for i in range(d):
		s += str(X[n][i]) + " "
	s += str(Y[n]) + "\n"
	F.write(s)