from collections import defaultdict
from math import sqrt
from operator import itemgetter
from collections import OrderedDict
from scipy.spatial import Delaunay
import numpy as np
import math
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


arr_ulaz = np.genfromtxt("ulaz.csv", delimiter=",")
arr_ulaz = arr_ulaz[1:]
arr_ulaz = arr_ulaz[arr_ulaz[:,0].argsort()]
arr = np.genfromtxt("zadnje2.csv", delimiter=' ',usecols=(0,1,2))
arr = arr[arr[:,0].argsort()]

fig, ax = plt.subplots()

for i in range(10000):
    circle1 = plt.Circle((arr[i][0],arr[i][1]), arr[i][2], color='r')
    circle2 = plt.Circle((arr_ulaz[i][0],arr_ulaz[i][1]),0.1,color = 'b')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
ax.set_xlim((0, 100))
ax.set_ylim((0, 100))

fig.savefig('plotcircles.png')
