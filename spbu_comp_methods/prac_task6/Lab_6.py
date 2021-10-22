#!/usr/bin/env python
# coding: utf-8

# In[557]:


import pandas as pd
import numpy as np
import math
from copy import copy
from numpy.linalg import norm
from scipy.linalg import hilbert, eig


# In[558]:


def max_abs(a): #максимальный по модулю в матрице элемент
    i_max,j_max = 0,1
    max_item = a[i_max,j_max]
    for i in range(a.shape[0]):
        for j in range(i+1, a.shape[0]):
            if abs(max_item) < abs(a[i,j]):
                max_item = a[i, j]
                i_max, j_max = i, j
    return i_max, j_max


# In[559]:


def jacobi_method(a,eps,strategy="circle"): #метод Якоби 
        iters = 0
        i,j = 0,0
        while True:
            h = np.identity(a.shape[0])
            if strategy == "abs":
                i,j = max_abs(a)
            else:
                if (j < (a.shape[0]-1) and j+1!=i):
                    j+=1
                elif j == a.shape[0]-1:
                    i+=1
                    j = 0
                else:
                    j+=2
            if i==a.shape[0]-1 and j==a.shape[0]:
                    return np.diag(a), iters
            if abs(a[i, j]) < eps:
                return np.diag(a), iters
            iters += 1
            phi = 0.5*(math.atan((2*a[i, j])/(a[i,i]-a[j,j])))
            c,s = math.cos(phi), math.sin(phi)
            h[i,i], h[j,j] = c,c
            h[i,j], h[j,i] = -s, s
            a = h.T@a@h


# In[560]:


def gersh_circles(a): #определение кругов Гершгорина
    ans = []
    for i in range(a.shape[0]):
        ans.append((a[i,i],sum(abs(a[i]))-abs(a[i,i])))
    return ans


# In[561]:


def is_in_circle(gersh,lmda): #проверка в принадлежности с.ч. хотя бы одному кругу
    return any([abs(c-lmda)<=r for c,r in gersh])


# In[562]:


X0 = np.array([[-5.509882,1.870086,0.422908],
              [0.287865,-11.811654,5.7119],
              [0.049099,4.308033,-12.970687]]) #матрица из учебника Н.В. Фаддевой и Д.К. Фаддеева


# In[563]:


matrixes = [X0,*[hilbert(n) for n in range(3,6)],hilbert(20)]


# In[564]:


X = pd.DataFrame(columns=['eps=10^(-2),res','eps=10^(-2),iters',
                          'eps=10^(-3),res', 'eps=10^(-3),iters',
                          'eps=10^(-4),res','eps=10^(-4),iters',
                         'eps=10^(-5),res','eps=10^(-5),iters'])

Y = pd.DataFrame(columns=['eps=10^(-2),res','eps=10^(-2),iters',
                          'eps=10^(-3),res', 'eps=10^(-3),iters',
                          'eps=10^(-4),res','eps=10^(-4),iters',
                         'eps=10^(-5),res','eps=10^(-5),iters'])

for matrix in matrixes:
    lambda_true = np.sort(eig(matrix)[0])
    row_X,row_Y = [],[]
    for i in range(2,6):
        lambda_abs,abs_iters = jacobi_method(matrix,10**(-i),strategy="abs")
        lambda_circle,circle_iters = jacobi_method(matrix,10**(-i),strategy="circle")
        row_X.extend([norm(np.sort(lambda_abs)-lambda_true),abs_iters])
        row_Y.extend([norm(np.sort(lambda_circle)-lambda_true),circle_iters])
    X = X.append(pd.Series(row_X,index=X.columns),True)
    Y = Y.append(pd.Series(row_Y,index=Y.columns),True)
    
X.index = ['X0','hilbert(3)','hilbert(4)','hilbert(5)','hilbert(20)']
Y.index = ['X0','hilbert(3)','hilbert(4)','hilbert(5)','hilbert(20)']


# In[565]:


X #стратетия с максимальным по модулю с.ч


# In[566]:


Y #стретия обнуления по порядку


# Проверим принадлежность найденных значений кругам Гершорина

# In[567]:


for matrix in matrixes:
    lambda_abs = jacobi_method(matrix,10**(-5),strategy="abs")[0]
    gersh = gersh_circles(matrix)
    print(all(([is_in_circle(gersh,lmbd) for lmbd in lambda_abs])))

