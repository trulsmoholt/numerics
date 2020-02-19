from math import sin, ceil
import numpy as np
import finitedifferences
from matplotlib import pyplot as plt
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits import mplot3d
def ploterror():
    gridSize = 5
    nTests = 7
    res = np.zeros([nTests,5])
    for i in range(0,nTests):
        res[i,0:2] = L2Error(gridSize,2*5**nTests)
        res[i,2] = 2/gridSize
        if(i>0):
            [res[i,3], intercept] = np.polyfit(np.log(res[i-1:i+1,2]),np.log(res[i-1:i+1,0]),1)
            [res[i,4], intercept] = np.polyfit(np.log(res[i-1:i+1,2]),np.log(res[i-1:i+1,1]),1)
        gridSize = gridSize * 2

    plt.subplot(121)
    plt.plot(res[1:,2],res[1:,3],"g^")
    plt.xlabel('steplength in space and time')
    plt.ylabel('Convergence RK4/centraldifference')

    plt.subplot(122)
    plt.plot(res[1:,2],res[1:,4],"g^")
    plt.xlabel('steplength in space and time')
    plt.ylabel('Convergence forward euler in time and space')
    plt.show()
    return
def plot3d(u,y,f):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    X,Y=np.meshgrid(np.linspace(0,1,m),np.linspace(0,T,n))
    ax.plot_surface(X, Y, u.T,cmap='viridis', edgecolor='none')

    ax.set_xlabel('X axis')
    ax.set_ylabel('t axis')
    ax.set_title("RK4/centraldifference")
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, y.T,cmap='viridis', edgecolor='none')
    ax.set_title("forward euler in time and space")
    ax.set_xlabel('X axis')
    ax.set_ylabel('t axis')
    plt.show()
    return

def plot1d(u,y,f):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.add_subplot(1, 2, 1)
    x = np.linspace(0,1,m,endpoint=False)
    plt.plot(x,u[:,u.shape[1]-1],x,f)
    plt.title("RK4/central difference")
    fig.add_subplot(1, 2, 2)
    plt.plot(x,y[:,y.shape[1]-1],x,f)
    plt.title("forward euler in time and space")
    plt.show()


def L2Error(m,n):
    x = np.linspace(0,1,m,endpoint = False)
    f = np.zeros(x.shape)
    f[ceil(m/3):ceil(2*m/3)] = 1
    u = np.zeros([m,n])
    y = u.copy()
    h = 1/m
    k = T/n
    A = finitedifferences.centralDifference(m)
    x = np.linspace(0,1,m)
    u = RK4(A,f,u,h,k)
    A = finitedifferences.forwardEuler(m)
    y = forwardEuler(A,f,y,h,k)

    u_error = np.abs(u[:,n-1]-f)
    y_error = np.abs(y[:,n-1]-f)
    return [np.dot(u_error,u_error)*h,np.dot(y_error,y_error)*h]

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


n = 60
m = 40
T = 1

h = 1/m
k = T/n
A = finitedifferences.centralDifference(m)
x = np.linspace(0,1,m,endpoint=False)
f = np.zeros(x.shape)
#f = np.sin(2*3.1415*x)
f[ceil(m/3):ceil(2*m/3)] = 1

u = np.zeros([m,n])
uE = uExc(u,T,k,x)
u = RK4(A,f,u,h,k)

A = finitedifferences.forwardEuler(m)
y = u.copy()
y = forwardEuler(A,f,y,h,k)
#plot1d(u,y,f)
error = np.abs(f-y[:,n-1])
ploterror()
print(np.dot(error,error)*h)



