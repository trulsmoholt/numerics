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



T=1
m=300
n = 3590
u = np.zeros([m,n])
h=1/m
k = T/n
A = finitedifferences.forwardEuler(m)
x = np.linspace(0,1,m)
f = np.sin(2*3.1415*x)
u[:,0] = f[:]
u = solver(A,f,u,h,k)
X,Y=np.meshgrid(np.linspace(0,1,m),np.linspace(0,T,n))
print(X.shape)
print(Y.shape)
print(u.shape)
print(k/h)
print(A)
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, u.T,cmap='viridis', edgecolor='none')
plt.show()