from math import sin
import numpy as np
import finitedifferences
from matplotlib import pyplot as plt
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import mplot3d



def solver(A,f,u,h,k):
    u[:,0]=f[:]
    for i in range(1,u.shape[1]):
        
        u[:,i]=u[:, i-1]+k/h*A @ u[:,i-1]

    return u



def RK4(A,f,u,h,k):
    u[:,0]=f[:]
    for i in range(1,u.shape[1]):
        k1 = k/h*A @ u[:,i-1]
        k2 = k/h*A @ (u[:,i-1] + 0.5*k1)
        k3 = k/h*A @ (u[:,i-1] + 0.5*k2)
        k4 = k/h*A @ (u[:,i-1] + k3)
        u[:,i] = u[:,i-1] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return u

def uExc(u,T,k,x):
    uExc = u.copy()
    for i in range(1,u.shape[1]):
        t=k*i
        uExc[:,i]=np.sin(2*3.1415*(x + t))
    return uExc





T=4
m=12
n = 200
u = np.zeros([m,n])
h=1/m
k = T/n
A = finitedifferences.centralDifference(m)


x = np.linspace(0,1,m)
f = np.sin(2*3.1415*x)
print(u.shape[1])
u = RK4(A,f,u,h,k)
X,Y=np.meshgrid(np.linspace(0,1,m),np.linspace(0,T,n))
print(X.shape)
print(Y.shape)
print(u.shape)
print(k/h)
print(A)

uE = uExc(u,T,k,x)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, u.T,cmap='viridis', edgecolor='none')

ax.set_xlabel('X axis')
ax.set_ylabel('t axis')
plt.show()