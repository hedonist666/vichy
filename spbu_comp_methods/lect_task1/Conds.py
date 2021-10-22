#!/usr/bin/env python
# coding: utf-8

# #  Сравнение чисел обусловленности обобщенной матрицы Вандермонда и ее корня

# In[34]:


from numpy.linalg import eig, inv, norm, cond
import numpy as np
from tabulate import tabulate


# In[2]:


def gen_vandermonde(n):
    result = np.zeros(shape=(n,n))
    for i in range(n):
        for k in range(n):
            result[i,k] = (n-i+1)**(-3*k/2)
    return result


# In[3]:


def newton_method(X,epsilon):
    x_k = np.identity(X.shape[0])
    x_k1 = 0.5*(x_k+inv(x_k)@X)
    while norm(x_k1-x_k)>epsilon:
        x_k = x_k1
        x_k1 = 0.5*(x_k+inv(x_k)@X)
    return x_k1


# In[9]:


def eig_method(X):
    V = eig(X)[1]
    sigma = np.diag(eig(X)[0]**(0.5))
    return V @ sigma @ inv(V)


# Числа обусловленности исходной матрицы X и ее корня B (найденных двумя методами):

# In[37]:


conds = []

for i in range(2,11):
    X = gen_vandermonde(i)
    B1 = eig_method(X)
    if i in range(2,8):
        B2 = newton_method(X,1e-3*(i**2))
    else:
        B2 = newton_method(X,1e-2*(i**2))
    conds.append([i,cond(X),cond(B1),cond(B2),norm(B2@B2-X)])

print(tabulate(conds,headers=['n','cond1','cond2','cond3','norm((X - NewtonMeth^2))'],
               tablefmt='github',numalign="right"))


# # Приложение:

# Обобщенная матрица Вандермонда второго порядка:

# In[18]:


X = gen_vandermonde(2)
print(X)


# Корень из обобщенной матрицы Вандермонда второго порядка:

# In[19]:


print(eig_method(X))


# Обобщенная матрица Вандермонда третьего порядка:

# In[20]:


X = gen_vandermonde(3)
print(X)


# Корень из обобщенной матрицы Вандермонда третьего порядка:

# In[21]:


print(eig_method(X))

