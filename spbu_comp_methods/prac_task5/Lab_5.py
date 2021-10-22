#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from numpy.linalg import eig
from scipy.linalg import hilbert
from copy import copy


# In[98]:


def pow_method(a,eps,x0=None): #степенной метод
    if x0 is None:
        x0 = np.random.uniform(-1,1,size=a.shape[1])
    x1 = a@x0
    num_of_iters = 1
    lambda0 = x1[0]/x0[0]
    while True:
        x0,x1 = x1, a@x1
        lambda1 = x1[0]/x0[0]
        if abs(lambda1-lambda0)<eps or num_of_iters > 5000:
            break
        lambda0 = lambda1
        num_of_iters += 1
    return abs(lambda1),num_of_iters


# In[99]:


def scal_method(a,eps,x0=None): #метод скалярных произведений
    if x0 is None:
            x0 = np.random.uniform(-1,1,size=a.shape[1])
    num_of_iters = 1
    x1 = a@x0
    y0 = copy(x0)
    a_T = np.transpose(a)
    y1 = a_T@x0
    lambda0 = np.dot(x1,y1)/np.dot(x0,y0)
    while True:
        x0,x1 = x1, a@x1
        y0,y1 = y1, a_T@y1
        lambda1 = np.dot(x1,y1)/np.dot(x0,y1)
        if abs(lambda1-lambda0)<eps or num_of_iters > 5000:
            break
        lambda0 = lambda1
        num_of_iters += 1
    return abs(lambda1),num_of_iters


# Матрицы из учебника Д.К. Фадддева и В.Н. Фаддеевой:

# In[100]:


X0 = np.array([[-5.509882,1.870086,0.422908],
              [0.287865,-11.811654,5.7119],
              [0.049099,4.308033,-12.970687]])
X1 = np.array([[4.2,-3.4,0.3],
              [4.7,-3.9,0.3],
              [-5.6,5.2,0.1]])


# In[101]:


matrixes = [X0,X1,hilbert(3),hilbert(4)]


# In[107]:


X = pd.DataFrame(columns=['eps=10^(-2),res','eps=10^(-2),iters',
                          'eps=10^(-3),res', 'eps=10^(-3),iters',
                          'eps=10^(-4),res','eps=10^(-4),iters',
                         'eps=10^(-5),res','eps=10^(-5),iters'])

Y = pd.DataFrame(columns=['eps=10^(-2),res','eps=10^(-2),iters',
                          'eps=10^(-3),res', 'eps=10^(-3),iters',
                          'eps=10^(-4),res','eps=10^(-4),iters',
                         'eps=10^(-5),res','eps=10^(-5),iters'])

for matrix in matrixes:
    x0 = np.ones(matrix.shape[1])
    lambda_true = max(abs(eig(matrix)[0]))
    row_X,row_Y = [],[]
    for i in range(2,6):
        lambda_pow,pow_iters = pow_method(matrix,10**(-i),x0)
        lambda_scal,scal_iters = scal_method(matrix,10**(-i),x0)
        row_X.extend([abs(lambda_pow-lambda_true),pow_iters])
        row_Y.extend([abs(lambda_scal-lambda_true),scal_iters])
    X = X.append(pd.Series(row_X,index=X.columns),True)
    Y = Y.append(pd.Series(row_Y,index=Y.columns),True)
    
X.index = ['X0','X1','hilbert(3)','hilbert(4)']
Y.index = ['X0','X1','hilbert(3)','hilbert(4)']


# In[108]:


X #результаты для степенного метола


# In[109]:


Y #результаты для метода скалярных произведений

