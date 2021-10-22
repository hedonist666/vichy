#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy.linalg import norm, cond, solve, inv
from scipy.linalg import hilbert


# In[2]:


def iterational_method(alpha,beta,x0,eps): #общая схема итерационного метода
    num_of_iters = 1
    x1 = alpha @ x0 + beta
    while (norm(x1-x0)>eps and num_of_iters < 500):
        x0 = x1
        x1 = alpha @ x0 + beta
        num_of_iters += 1
    return x1, num_of_iters


# In[3]:


def simple_iterational_method(a,b,x0,eps): #метод простых итераций
    alpha = np.zeros([a.shape[0],a.shape[1]])
    beta = np.zeros(b.shape[0])
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if i != j:
                alpha[i,j] = -a[i,j]/a[i,i]
                beta[i] = b[i]/a[i,i]
    return iterational_method(alpha,beta,x0,eps)


# In[4]:


def seidel_method(a,b,x0,eps): #метод Зейделя
    n, m = a.shape[0], a.shape[1]
    l,r,d = [np.zeros([n,m]) for _ in range(3)]
    for i in range(n):
        for j in range(m):
            if i > j:
                l[i,j] = a[i,j]
            elif i < j:
                r[i,j] = a[i,j]
            else:
                d[i,j] = a[i,j]
    beta = inv(d+l)
    return iterational_method(-beta@r,beta@x0,x0,eps)


# Матрицы из методички А.Н. Пакулиной:

# In[5]:


X1 = np.array([[-401.64, 200.12], 
               [21200.72,  -601.76]]) 

X2 = np.array([[-400.94, 200.02],
               [1200.12, -600.96]])


# In[7]:


def diag_dominant_matrix(a_ii,n): 
    ans = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                ans[i,j] = a_ii
            if i==j-1 or i==j+1:
                ans[i,j] = -1
    return ans


# In[24]:


matrixes = [X1,X2,diag_dominant_matrix(4,3),diag_dominant_matrix(2,5),hilbert(3),hilbert(4),hilbert(5)]


# In[25]:


X = pd.DataFrame(columns=['eps=10^(-4),||x_eps-x||','eps=10^(-4),iters',
                          'eps=10^(-8),||x_eps-x||', 'eps=10^(-8),iters',
                          'eps=10^(-12),||x_eps-x||','eps=10^(-12),iters'])

Y = pd.DataFrame(columns=['eps=10^(-4),||x_eps-x||','eps=10^(-4),iters',
                          'eps=10^(-8),||x_eps-x||', 'eps=10^(-8),iters',
                          'eps=10^(-12),||x_eps-x||','eps=10^(-12),iters'])


for matrix in matrixes:
    b = np.random.uniform(-100,100,size=matrix.shape[0])
    x = solve(matrix,b)
    row_X,row_Y = [],[]
    for i in range(4,13,4):
        x_seidel,seidel_iters = seidel_method(matrix,b,b,10**(-i))
        x_iter,iter_iters = simple_iterational_method(matrix,b,b,10**(-i))
        row_X.extend([norm(x_seidel-x),seidel_iters])
        row_Y.extend([norm(x_iter-x),iter_iters])
    X = X.append(pd.Series(row_X,index=X.columns),True)
    Y = Y.append(pd.Series(row_Y,index=Y.columns),True)
    
X.index = ['X1','X2','diag_dominant_matrix(4,3)','diag_dominant_matrix(2,5)','hilbert(3)','hilbert(4)','hilbert(5)']
Y.index = ['X1','X2','diag_dominant_matrix(4,3)','diag_dominant_matrix(2,5)','hilbert(3)','hilbert(4)','hilbert(5)']


# In[26]:


X #результаты для метода Зейделя


# In[27]:


Y #результаты для метода простой итерации

