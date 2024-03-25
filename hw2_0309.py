# -*- coding: utf-8 -*-
"""HW2_0309.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rRFUd1ak259Mu84fD3BBvrQNzjqid5S8
"""

import numpy as np

def estimate_volume(n, N):
    count = 0
    for i in range(N):
        x = np.random.uniform(low=-1, high=1, size=n)
        if np.linalg.norm(x) <= 1:
            count += 1
    volume_estimate = (2**n) * count / N
    print("Volume of hypersphere:", volume_estimate)
    return volume_estimate

vol_list = []
times = 10
n = 5
N = 1000000
for i in range(times):
  vol = estimate_volume(n, N)
  vol_list.append(vol)
print(vol_list)
print('avg_vol = {}'.format(np.average(vol_list)))
print('std_vol = {}'.format(np.std(vol_list)))

"""HW2第一題"""

import numpy as np
x1=0
x2=1
x3=3
X = np.array([
    [1,x1,x1*x1],
    [1,x2,x2*x2],
    [1,x3,x3*x3]
    ])
(x1,y1)=(0,0.5)
(x2,y2)=(1,2.5)
(x3,y3)=(3,12.5)
A = np.transpose(X)
print(A.dot(X))
y = np.array ([
    [0.5],
    [2.5],
    [12.5]
    ])
#A_inv = np.linalg.inv(A)
B = np.linalg.inv(A.dot(X)).dot(A).dot(y)
print(B)

"""HW2第三題

# **`Problem 1`**
"""

import random
def cal_hypersphere_vol(dim,count):
  n = dim # number of dimensions
  num_points = count # number of random points to generate

  count_inside = 0 # count of points inside hypersphere

  for i in range(num_points):
      point = [random.uniform(-1, 1) for j in range(n)] # generate random point
      distance = sum([x**2 for x in point]) ** 0.5 # calculate distance from origin
      if distance <= 1: # check if point is inside hypersphere
          count_inside += 1

  proportion = count_inside / num_points # calculate proportion of points inside hypersphere
  volume = proportion * 2**n # calculate volume of hypersphere

  print("Volume of hypersphere:", volume)
  return volume

import numpy as np
dim = 5
count = 1000000
times = 10
vol_list = []
for i in range(times):
  vol = cal_hypersphere_vol(dim,count)
  vol_list.append(vol)
print(vol_list)
print('avg_vol = {}'.format(np.average(vol_list)))
print('std_vol = {}'.format(np.std(vol_list)))

"""# **problem2**"""

import numpy as np
import matplotlib.pyplot as plt
import collections

# Define the inequality
def inequality(x, y):
  return (9*x + 4*y) <= 49 and x >= 0 and x<= 5 and y >=0 and y<= 10

def is_corner(x, y):
  outcome = [abs((9*x + 4*y)-49)<0.1,x == 0,x== 5,y ==0,y== 10]
  return collections.Counter(outcome)[True] == 2

def target(x,y,z):
  if len(z) and x+y > z[0][1]:
    z[0] = ([x,y],x+y)
  else:
    z.append(([x,y],x+y))
  return z

# Create a grid of x and y values
X = np.linspace(0, 5, 1000)
Y = np.linspace(0, 10, 1000)

x = []
y = []
corner_list = []
z = []
for i in X:
  for j in Y:
    if inequality(i, j):
      z = target(i,j,z)

      x.append(i)
      y.append(j)
      if is_corner(i, j):
        corner_list.append([i,j])
plt.scatter(x,y,c = 'gray')
for vertex in corner_list:
  plt.scatter(vertex[0], vertex[1], marker = '*', c = 'red')
plt.show()
print('Max Z value is {} and the point is {}'.format(z[0][1],z[0][0]))