# -*- coding: utf-8 -*-
"""ML_HW5

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EW5dnEXoMljODz_eInBNgGmYERYbb3uk
"""

import random
import matplotlib.pyplot as plt
import math
x_set=[]
r_set=[]
y_set=[]
w1_set=[]
w0_set=[]
eta=0.01
w1=0.01
w0=0.01

def y(x):
  if x<1:
    return 0
  elif x>1 and x<3:
    return 0.5*(x+1)
  else:
    return 1
def sigmoid(x,w1,w0):
  return 1/(1+math.exp(-w1*x-w0))
for intersation in range(30):
  x=random.uniform(-4,4)
  zeta=random.uniform(0,1)
  if zeta>0 and zeta<y(x):
    x_set.append(float(x))
    r_set.append(1)
    y_set.append(y(x))
  else:
    x_set.append(float(x))
    r_set.append(0)
    y_set.append(y(x))
print(x_set)

epoch=100
sum1=0
sum2=0
sum_en=0
for intersation2 in range(epoch):
  for i in range(30):
    z=sigmoid(x_set[i],w1,w0)
    sum1=sum1+(r_set[i]-z)*x_set[i]
    sum2=sum2+(r_set[i]-z)
    sum_en=sum_en-(r_set[i]*math.log(z)+(1-r_set[i])*math.log(1-z))

  w1=w1+eta*sum1
  w0=w0+eta*sum2
  w1_set.append(w1)
  w0_set.append(w0)
print(w1_set)
print(w0_set)

import numpy as np

# Define parameters of Gaussians
mu1 = np.array([0, 0])
mu2 = np.array([3, 4])
error1=0
error2=0
midpoint = (mu1 + mu2) / 2
r = np.linalg.norm(mu1 - midpoint)

# Classify samples from Gaussian 1
misclassified = 0
for i in range(1000):
    x1=random.uniform(0,1)
    x2=random.uniform(0,1)
    x=np.array([x1,x2])
    if np.linalg.norm(x - midpoint) > r:
      error1 += 1
    else:
      pass
for i in range(1000):
    if np.linalg.norm(x - midpoint) > r:
      error2 += 1
    else:
      pass











# Example
import random
import math
w1_set = 1.2
w0_set = 0
x = 0.0
outcome = 0
w1 = 0.01
w0 = 0.01
eta = 0.01
x_record = []
r_record = []

def sigmod(x, w1, w0):
    return 1/(1 + math.exp(-w1 * x - w0))

for iteration in range(30):
    x = random.uniform(-2,2)
    zeta = random.uniform(0,1)
    y = sigmod(x, w1_set, w0_set)
    if zeta <= y:
        outcome = 1
    else:
        outcome = 0
    x_record.append(float(x))
    r_record.append(float(outcome))
    print('iteration : {}, x : {}, {}'.format(iteration, round(x,2), outcome))
epoch = 100
sum1 = 0.0
sum2 = 0.0
sum_en = 0.0
print('sum1 = {}, sum2 = {}'.format(sum1, sum2))
for iteration2 in range(epoch):
    sum2 = 0.0
    sum1 = 0.0
    sum_en = 0.0
    for i in range(30):
        y = sigmod(x_record[i],w1,w0)
        sum1 = sum1 + (r_record[i] - y) * x_record[i]
        sum2 = sum2 + (r_record[i] - y)
        #computing cross entropy
        sum_en = sum_en - (r_record[i] * math.log(y) + (1 - r_record[i]) * math.log(1 - y))
    w1 = w1 + eta * sum1
    w0 = w0 + eta * sum2
    if iteration2 % 10 == 0:
        print('iteration : {}, w1 : {}, w0 : {}, sum1 : {}, sum2 : {}, sum_en : {}'.format(iteration2, round(w1,2), round(w0,2), round(sum1,2), round(sum2,2), round(sum_en,2)))
print('final_result: w1 : {}, w0 : {}'.format(round(w1,2), round(w0,2)))





"""#**`HW1`**"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt

#initial parameters
w1_set = 1.2
w0_set = 0
x = 0.0
outcome = 0
iteration_time = 30
w1 = 0.01
w0 = 0.01
eta = 0.01
x_record = []
r_record = []

#function
def sigmod(x, w1, w0):
    return 1/(1 + math.exp(-w1 * x - w0))

def piecewise(x):
    if x < 1:
        return 0
    elif x > 1 and x < 3:
        return 0.5 * (x + 1)
    elif x > 3:
        return 1
    else:
        return None

#
for iteration in range(iteration_time):
    x = random.uniform(-4,4)
    zeta = random.uniform(0,1)
    y = sigmod(x, w1_set, w0_set)
    outcome = piecewise(y)
    x_record.append(float(x))
    r_record.append(float(outcome))
    print('iteration : {}, x : {}, {}'.format(iteration, round(x,2), outcome))
#(a) Plot the new data for the interval [-4,4]
plt.scatter(range(0,30),x_record)
plt.title('x_record')
plt.show()

epoch = 100
sum1 = 0.0
sum2 = 0.0
sum_en = 0.0
sum_en_list = []
print('sum1 = {}, sum2 = {}'.format(sum1, sum2))

for iteration2 in range(epoch):
    sum2 = 0.0
    sum1 = 0.0
    sum_en = 0.0
    for i in range(30):
        y = sigmod(x_record[i],w1,w0)
        sum1 = sum1 + (r_record[i] - y) * x_record[i]
        sum2 = sum2 + (r_record[i] - y)
        #computing cross entropy
        sum_en = sum_en - (r_record[i] * math.log(y) + (1 - r_record[i]) * math.log(1 - y))
    sum_en_list.append(sum_en)
    w1 = w1 + eta * sum1
    w0 = w0 + eta * sum2
    if iteration2 % 10 == 0:
        print('iteration : {}, w1 : {}, w0 : {}, sum1 : {}, sum2 : {}, sum_en : {}'.format(iteration2, round(w1,2), round(w0,2), round(sum1,2), round(sum2,2), round(sum_en,2)))
print('final_result: w1 : {}, w0 : {}'.format(round(w1,2), round(w0,2)))
# Plot the logistic function using w1 and w0
plt.scatter(x_record,1/(1 + np.exp(-np.array([w1]) * x_record - np.array([w0]))))
plt.title('logistic function using w1 and w0')
plt.show()
# Plot the cross entropy versus number of iteration

plt.plot(sum_en_list)
plt.title('cross entropy')
plt.show()

"""#**`HW2`**"""



import random
x1_record = []
x2_record = []
miu_x = np.array([0,0])
miu_y = np.array([3,4])
sigma_x = 1
sigma_y = 3
x1 = 0.0
x2 = 0.0
y1 = 0.0
y2 = 0.0
count1_error = 0
count2_error = 0
sample_times =1000
experiment_times = 10
error_rate = []

#calculate x_c and radius
x_c = (sigma_y**2 * miu_x - sigma_x**2 * miu_y)/ (sigma_y - sigma_x)
r = ((sigma_x**2 * sigma_y**2)/(sigma_y**2 - sigma_x**2) * (((np.linalg.norm(miu_x + miu_y))**2 / (sigma_y**2 - sigma_x**2)) + 4 * np.log(sigma_y/sigma_x)))**(1/2)
print('x_c = {}, r = {}'.format(x_c, r))

for experiment_time in range(experiment_times):
    #decision boundary
    for sample_time in range(sample_times):
        x1 = random.gauss(0,sigma_x)
        x2 = random.gauss(0,sigma_x)
        x1_record.append(x1)
        x2_record.append(x2)
        if (x1 - x_c[0])**2 + (x2 - x_c[1])**2 > r**2:
            count1_error = count1_error + 1
        else:
            pass

        y1 = 3.0 + random.gauss(0,sigma_y)
        y2 = 4.0 + random.gauss(0,sigma_y)
        if (y1 - x_c[0])**2 + (y2 - x_c[1])**2 < r**2:
            count2_error = count2_error + 1
        else:
            pass
    print('experiment_time : {}'.format(experiment_time + 1))
    print('error_rate1 = {}%'.format(count1_error/sample_times * 100))
    print('error_rate2 = {}%'.format(count2_error/sample_times * 100))
    error_rate.append(count2_error/sample_times)
print('average error rate = {}% and std = {}'.format(sum(error_rate)/experiment_times * 100, np.std(error_rate)))

