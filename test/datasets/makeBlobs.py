from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
n = 300000
X, y = make_blobs(n_samples=n, centers=500, n_features=2, random_state=1)

F = open("BigData_"+str(n)+".txt", "w")
F = open("BigData_"+str(n)+".txt", "a")
for n in range(X.shape[0]):
    s = str(X[n][0])+" "+str(X[n][1])+" "+str(y[n])+"\n"
    F.writelines(s)


# for n in range(X.shape[0]):
 #   plt.scatter(X[n][0], X[n][1])


plt.show()
