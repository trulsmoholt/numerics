from math import sin
import numpy as np
import finitedifferences
from matplotlib import pyplot as plt
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import mplot3d

PI = 3.1415

def uExc(u,T,k,x):
    uExc = u.copy()
    for i in range(1,u.shape[1]):
        t=k*i
        uExc[:,i]=np.sin(2*3.1415*(x + t))
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

def L2Error(m,n,T):
    u = np.zeros([m,n])
    h = 1/m
    k = T/n
    A = finitedifferences.centralDifference(m)
    x = np.linspace(0,1,m,endpoint= False)
    f = np.sin(2*PI*x)
    u = RK4(A,f,u,h,k)
    #uE = uExc(u,T,k,x)

    error = np.abs(u[:,n-1]-f)
    return np.dot(error,error)*h
nTests = 7
gridSize = 5
res = np.zeros([nTests,4])
for i in range(0,nTests):
    res[i,0] = L2Error(gridSize,5*2*2*2*2**nTests,2)
    res[i,1] = 2/gridSize
    if(i>0):
        [res[i,2], intercept] = np.polyfit(np.log(res[i-1:i+1,1]),np.log(res[i-1:i+1,0]),1)
    gridSize = gridSize * 2

[slope, intercept] = np.polyfit(np.log(res[:,1]),np.log(res[:,0]),1)
print(res)

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(121)
plt.plot(res[:,1],res[:,0],"bs")
plt.xlabel('steplength in space and time')
plt.ylabel('L2 error')
plt.gca().invert_xaxis()
plt.subplot(122)
plt.plot(res[1:,1],res[1:,2],"g^")
plt.xlabel('steplength in space and time')
plt.ylabel('Convergence/slope of loglog plot')
plt.gca().invert_xaxis()
plt.savefig("loglog_slope.png")
plt.show()





