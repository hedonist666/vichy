#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy.linalg import solve, inv, det, norm, cond
from scipy.linalg import hilbert


# In[2]:


def spectrum_criterion(matrix): #спектральный критерий
    return norm(matrix)*norm(inv(matrix))


# In[3]:


def volume_criterion(matrix): #критерий Ортеги
    vol = 1
    for n in range(matrix.shape[0]):
        vol *= norm(matrix[n])
    return vol/abs(det(matrix))


# In[4]:


def angle_criterion(matrix): #угловой критерий
    C = inv(matrix)
    return max([norm(a_n)*norm(c_n) for a_n, c_n in zip(matrix,np.transpose(C))])


# In[5]:


def conds(matrix):
    return(spectrum_criterion(matrix),volume_criterion(matrix),angle_criterion(matrix))


# In[6]:


def diag_dominant_matrix(a_ii,n):
    ans = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                ans[i,j] = a_ii
            if i==j-1 or i==j+1:
                ans[i,j] = -1
    return ans


# Матрицы из методички А.Н. Пакулиной:

# In[7]:


X1 = np.array([[-401.64, 200.12], 
               [21200.72,  -601.76]]) 

X2 = np.array([[-400.94, 200.02],
               [1200.12, -600.96]])


# Свои матрицы:

# In[8]:


X3 = np.array([[3,4],
              [4,1]])

X4 = np.array([
    [0.1,0.2,0.3],
    [0.1,0.1,0.1],
    [0.4,0.9,0.7]])


# In[9]:


matrixes = [X1,X2,diag_dominant_matrix(4,3),diag_dominant_matrix(2,5),hilbert(4),hilbert(5),X3,X4]


# In[37]:


def solution(matrix):
    res = []
    b = np.random.uniform(-100,100,size=matrix.shape[0])
    sol = solve(matrix,b)
    for i in (-2,-5,-8):
        res.append(norm(solve(matrix-10**(i),b - 10**(i))-sol))
    return sol,res


# In[38]:


X = pd.DataFrame(columns=['Спектральный критерий','Критерий Ортеги','Угловой критерий','eps=10^(-2)','eps=10^(-5)','eps=10^(-8)'])

for matrix in matrixes:
    row = []
    row.extend(conds(matrix))
    row.extend(solution(matrix)[1])
    row_series = pd.Series(row,index=X.columns)
    X = X.append(row_series,True)
    
X.index = ['X1','X2','diag_dominant_matrix(4,3)','diag_dominant_matrix(2,5)','hilbert(4)','hilbert(5)','X3','X4']

X

