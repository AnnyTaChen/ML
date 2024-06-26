# -*- coding: utf-8 -*-
"""ML_hw3

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kTecvyiHywg-ihArinzxdswXyTPJ5scM
"""

import matplotlib.pyplot as plt
import random
import numpy as np
P_initialA=1/3
P_initialB=1/3
P_initialC=1/3

P_red_A=7/10
P_green_A=3/10
P_red_B=3/10
P_green_B=7/10
P_red_C=1/10
P_green_C=9/10

P_A_red=0
P_A_green=0
P_B_red=0
P_B_green=0
P_C_red=0
P_C_green=0
num=100
x_array=np.array([])
y_array=np.array([])
z_array=np.array([])
for i in range(num):
  x=random.uniform(0,1)
  if x>=0 and x<=P_green_C:
     P_C_green=P_green_C*P_initialC /(P_green_A*P_initialA+P_green_B*P_initialB+P_green_C*P_initialC)
     P_B_green=P_green_B*P_initialB /(P_green_A*P_initialA+P_green_B*P_initialB+P_green_C*P_initialC)
     P_A_green=P_green_A*P_initialA /(P_green_A*P_initialA+P_green_B*P_initialB+P_green_C*P_initialC)
     P_initialA=P_A_green
     P_initialB=P_B_green
     P_initialC=P_C_green
     x_array = np.append(x_array,round(P_A_green,4))
     y_array = np.append(y_array,round(P_B_green,4))
     z_array = np.append(z_array,round(P_C_green,4))
  else:
     P_C_red=P_red_C*P_initialC /(P_red_A*P_initialA+P_red_B*P_initialB+P_red_C*P_initialC)
     P_B_red=P_red_B*P_initialB /(P_red_A*P_initialA+P_red_B*P_initialB+P_red_C*P_initialC)
     P_A_red=P_red_A*P_initialA /(P_red_A*P_initialA+P_red_B*P_initialB+P_red_C*P_initialC)
     P_initialA=P_A_red
     P_initialB=P_B_red
     P_initialC=P_C_red
     x_array = np.append(x_array,round(P_A_red,4))
     y_array = np.append(y_array,round(P_B_red,4))
     z_array = np.append(z_array,round(P_C_red,4))
for j in range(num):
  print((x_array[j]),'\t')
print('--------------------------------------------------')
for j in range(num):
  print((y_array[j]),'\t')
print('--------------------------------------------------')
for j in range(num):
  print((z_array[j]),'\t')



import numpy as np
import random
num=10**6
for i in range(10):
  z=1000
  print('第',i+1,'次')
  for j in range(num):
    w1=random.uniform(0,1)
    w2=random.uniform(0,1)
    w3=random.uniform(0,1)
    b=np.array([[3],[1/2],[1/2]])
    w=np.array([[w1],[w2],[w3]])
    zFinal=np.dot((w-b).T,(w-b))
    if zFinal<z:
      z=zFinal
      w1_min=w1
      w2_min=w2
      w3_min=w3
  print('z=',z)
  print('w1=',w1_min,'\nw2=',w2_min,'\nw3=',w3_min)

import numpy as np
import random
num=10**6
for i in range(10):
  z=1000
  print('第',i+1,'次')
  for j in range(num):
    w1=random.uniform(0,1)
    w2=random.uniform(0,1)
    w3=random.uniform(0,1)
    w4=random.uniform(0,1)
    w5=random.uniform(0,1)
    b=np.array([[3],[1/2],[1/2],[1/2],[1/2]])
    w=np.array([[w1],[w2],[w3],[w4],[w5]])
    zFinal=np.dot((w-b).T,(w-b))
    if zFinal<z:
      z=zFinal
      w1_min=w1
      w2_min=w2
      w3_min=w3
      w4_min=w4
      w5_min=w5
  print('z=',z)
  print('w1=',w1_min,'\nw2=',w2_min,'\nw3=',w3_min,'\nw4=',w4_min,'\nw5=',w5_min)