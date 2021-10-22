#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy.linalg import solve, norm, cond
from scipy.linalg import hilbert


# In[2]:


def lu(a): #алгоритм LU-декомпозиции
    n = a.shape[0]
    l = np.identity(n)
    u = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i <= j:
                u[i,j] = a[i,j]-sum([l[i,k]*u[k,j] for k in range(i)])
            else:
                l[i,j] = (a[i,j]-sum([l[i,k]*u[k,j] for k in range(j)]))/u[j,j]
    return l,u


# In[3]:


def lu_solve(l,u,b=None): #решение СЛАУ LU-методом
    if b is None:
        b = np.random.uniform(-100,100,size=(u.shape[1]))
    y = np.zeros(l.shape[1])
    x = np.zeros(u.shape[1])
    n = len(x)
    for i in range(len(y)):
        y[i] = b[i] - sum([l[i,p]*y[p] for p in range(i)])
    for j in range(len(x)):
        x[n-j-1]=(y[n-j-1]-sum([u[n-j-1,n-p-1]*x[n-p-1] for p in range(j)]))/u[n-j-1,n-j-1]
    return x
            


# Проверим, что решения, полученные lu-методом, совпадают с решением системы:

# In[13]:


a = np.random.rand(10,10)
b = np.random.rand(10)
l,u = lu(a)
norm(lu_solve(l,u,b)-solve(a,b))


# In[5]:


matrixes = [hilbert(n) for n in range(3,6)]


# In[8]:


def regularisation_solution(a,b=None):
    if b is None:
        b = np.random.uniform(-100,100,size=(a.shape[1]))
    ans = pd.DataFrame(columns=["alpha","cond(a+alpha*E)","||x-x_alpha||","||b-A*x_alpha||"])
    l,u = lu(a)
    x = lu_solve(l,u,b)
    ans = ans.append(pd.Series([0,cond(a),x,norm(b-a@x)],index=ans.columns),True)
    E = np.identity(a.shape[0])
    x = solve(a,b)
    for i in range(2,13,2):
        a_i = a + 10**(-i)*E
        l,u = lu(a_i)
        x_i = lu_solve(l,u,b)
        ans = ans.append(pd.Series([10**(-i),cond(a_i),norm(x_i-x),norm(b-a@x_i)],index=ans.columns),True)
    return ans


# In[9]:


regularisation_solution(matrixes[0],np.array([1,1,1])) #результат для матрицы Гильберта 3-го порядка


# In[21]:


regularisation_solution(matrixes[1],np.array([1,1,1,1])) #результат для матрицы Гильберта 4-го порядка


# In[22]:


regularisation_solution(hilbert(5),hilbert(5)@np.ones(5)) #результат для матрицы Гильберта 5-го порядка


# In[ ]:




