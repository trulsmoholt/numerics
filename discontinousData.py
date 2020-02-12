from math import sin, ceil
import numpy as np
import finitedifferences
from matplotlib import pyplot as plt
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import mplot3d

def uExc(u,T,k,x):
    uExc = u.copy()
    [m,n] = uExc.shape
    for i in range(1,u.shape[1]):
        t=k*i
        bnd = (1/2+t)
        uExc[:ceil(m*bnd),i]=1
    return uExc

def RK4(A,f,u,h,k):
    u[:,0]=f[:]
    for i in range(1,u.shape[1]):
        k1 = k/h*A @ u[:,i-1]
        k2 = k/h*A @ (u[:,i-1] + 0.5*k1)
        k3 = k/h*A @ (u[:,i-1] + 0.5*k2)
        k4 = k/h*A @ (u[:,i-1] + k3)
        u[:,i] = u[:,i-1] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return u

def forwardEuler(A,f,u,h,k):
    u[:,0]=f[:]
    for i in range(1,u.shape[1]):
        u[:,i] = u[:,i-1] + k/h*A @ u[:,i-1]
    return u


n = 15
m = 15
T = 1/2

u = np.zeros([m,n])
h = 1/m
k = T/n
A = finitedifferences.centralDifference(m)
x = np.linspace(0,1,m)
f = np.zeros(x.shape)
f[:ceil(m/2)] = 1

uE = uExc(u,T,k,x)
u = RK4(A,f,u,h,k)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')

X,Y=np.meshgrid(np.linspace(0,1,m),np.linspace(0,T,n))
ax.plot_surface(X, Y, u.T,cmap='viridis', edgecolor='none')

ax.set_xlabel('X axis')
ax.set_ylabel('t axis')
ax.set_title("RK4/centraldifference")
ax = fig.add_subplot(1, 2, 2, projection='3d')
A = finitedifferences.forwardEuler(m)
u = RK4(A,f,u,h,k)
ax.plot_surface(X, Y, u.T,cmap='viridis', edgecolor='none')
ax.set_title("forward euler in time and space")
ax.set_xlabel('X axis')
ax.set_ylabel('t axis')
plt.show()

plt.show()