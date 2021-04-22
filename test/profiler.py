import subprocess
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import math

mpl.use('Agg')

ks = [int(2**x) for x in range(1, 10)]
ns = [int(10**x) for x in range(6, 1, -1)]

speedups = []

min_su = math.inf
max_su = -math.inf
for n in ns:
    su = []
    for k in ks:
        cmd = "./src/test datasets/BigData_" + str(n) + ".txt kmeans " + str(k) + " 10" 
        print(cmd)
        sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        subprocess_return = str(sp.stdout.read())
        speedup_cur = float(subprocess_return.split("Obtained: ")[1].split("x")[0])
        print(n, k, speedup_cur)
        su.append(speedup_cur)
        min_su = min(min_su, speedup_cur)
        max_su = max(max_su, speedup_cur)
    speedups.append(su)

print(speedups)

fig, ax = plt.subplots(1, 1)
ax.set_ylabel("Number of samples (n) (log scale)")
ax.set_xlabel("Number of clusters (k) (log scale)")
print("Plotting")
img = ax.imshow(speedups, cmap='hot', norm=mpl.colors.LogNorm(vmin=min_su, vmax=max_su))
fig.colorbar(img)
ax.set_xticks(np.arange(len(ks)))
ax.set_yticks(np.arange(len(ns)))
ax.set_xticklabels([str(k) for k in ks])
ax.set_yticklabels([str(n) for n in ns])
plt.title("Kmeans speedup (log) as a function of k and n")
plt.tight_layout()
plt.savefig('kmeans.png')

##################################################################################################
# 
# rs = [float(10**x) for x in range(-3, 4)]
# mps = [int(2**x) for x in range(1, 8)]
# ns = [int(2**x) for x in range(7, 15)]
# print("rs:", rs)
# print("mps:", mps)
# print("ns:", ns)
# 
# 
# speedups_fixed_n = []
# 
# min_su = math.inf
# max_su = -math.inf
# for r in rs:
#     su = []
#     for mp in mps:
#         cmd = "./src/test datasets/BigData_" + str(ns[-1]) + ".txt dbscan " + str(mp) + " " + str(r) 
#         print(cmd)
#         sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
#         subprocess_return = str(sp.stdout.read())
#         print(subprocess_return)
#         speedup_cur = float(subprocess_return.split("Obtained: ")[1].split("x")[0])
#         print(r, mp, speedup_cur)
#         su.append(speedup_cur)
#         min_su = min(min_su, speedup_cur)
#         max_su = max(max_su, speedup_cur)
#     speedups_fixed_n.append(su)
# 
# print(speedups_fixed_n)
# 
# fig, ax = plt.subplots(1, 1)
# ax.set_xlabel("MinPoints (mpts) (log scale)")
# ax.set_ylabel("Radius (r) (log scale)")
# print("Plotting fixed n")
# img = ax.imshow(speedups_fixed_n, cmap='hot')
# fig.colorbar(img)
# ax.set_xticks(np.arange(len(mps)))
# ax.set_yticks(np.arange(len(rs)))
# ax.set_xticklabels([str(mp) for mp in mps])
# ax.set_yticklabels([str(r) for r in rs])
# plt.title("DBScan speedup (log) as a function of\n mpts and r at number of samples = " + str(ns[-1]))
# plt.savefig('dbscan_fixed_n.png')
#################################
# 
# speedups_fixed_mp = []
# 
# min_su = math.inf
# max_su = -math.inf
# for r in rs:
#     su = []
#     for n in ns:
#         cmd = "./src/test datasets/BigData_" + str(n) + ".txt dbscan " + str(mps[-1])  + " " + str(r) 
#         print(cmd)
#         sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
#         subprocess_return = str(sp.stdout.read())
#         print(subprocess_return)
#         speedup_cur = float(subprocess_return.split("Obtained: ")[1].split("x")[0])
#         print(r, n, speedup_cur)
#         su.append(speedup_cur)
#         min_su = min(min_su, speedup_cur)
#         max_su = max(max_su, speedup_cur)
#     speedups_fixed_mp.append(su)
# 
# print(speedups_fixed_mp)
# 
# fig, ax = plt.subplots(1, 1)
# ax.set_xlabel("Number of Samples (n) (log scale)")
# ax.set_ylabel("Radius (r) (log scale)")
# print("Plotting fixed mp")
# img = ax.imshow(speedups_fixed_mp, cmap='hot')
# fig.colorbar(img)
# ax.set_xticks(np.arange(len(ns)))
# ax.set_yticks(np.arange(len(rs)))
# ax.set_xticklabels([str(n) for n in ns])
# ax.set_yticklabels([str(r) for r in rs])
# plt.title("DBScan speedup (log) as a function of n and r at minPts = "  + str(mps[-1]))
# plt.savefig('dbscan_fixed_mp.png')
# ############################
# 
# speedups_fixed_r = []
# 
# min_su = math.inf
# max_su = -math.inf
# for mp in mps:
#     su = []
#     for n in ns:
#         cmd = "./src/test datasets/BigData_" + str(n) + ".txt dbscan " + str(mp) + " " + str(10) 
#         print(cmd)
#         sp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
#         subprocess_return = str(sp.stdout.read())
#         print(subprocess_return)
#         speedup_cur = float(subprocess_return.split("Obtained: ")[1].split("x")[0])
#         print(n, mp, speedup_cur)
#         su.append(speedup_cur)
#         min_su = min(min_su, speedup_cur)
#         max_su = max(max_su, speedup_cur)
#     speedups_fixed_r.append(su)
# 
# print(speedups_fixed_r)
# 
# fig, ax = plt.subplots(1, 1)
# ax.set_xlabel("Number of Samples (n) (log scale)")
# ax.set_ylabel("Min points (mps) (log scale)")
# print("Plotting fixed r")
# img = ax.imshow(speedups_fixed_r, cmap='hot')
# fig.colorbar(img)
# ax.set_xticks(np.arange(len(ns)))
# ax.set_yticks(np.arange(len(mps)))
# ax.set_xticklabels([str(n) for n in ns])
# ax.set_yticklabels([str(mp) for mp in mps])
# plt.title("DBScan speedup (log) as a function of mps and n at radius = " + str(10))
# plt.savefig('dbscan_fixed_r.png')
# ############################

