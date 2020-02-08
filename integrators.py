import numpy as np
import matplotlib.pyplot as plt
x0 = np.array([[1],
                [-1]])

step = 0.01
t_span = np.arange(0,9,step)
print(type(t_span))
print(type(x0))
x_span = np.zeros((x0.shape[0],t_span.shape[0]))
print(x_span[:,2].shape)

x_span[:,0] = x0[:,0]
print(x_span)
A = [[1,-0.5],[-0.5,-1]]
for i in range(1,t_span.shape[0],1):
    x_span[:,i] = x_span[:,i-1]+step*np.matmul(A,x_span[:,i-1])
print(x_span)
plt.plot(t_span,x_span[0,:],t_span,x_span[1,:])
plt.show()