import numpy as np
import matplotlib.pyplot as plt
x0 = 0
def fun(x):
    return 2*x

step = 0.1
t_span = np.arange(0,2,step)
x_span = np.zeros(t_span.shape)

x_span[0] = x0
for i in range(1,t_span.shape[0],1):
    x_span[i] = x_span[i-1]+step*fun(t_span[i])
print(x_span)
plt.plot(t_span[0:19,0],x_span[0:19,0])
plt.show()