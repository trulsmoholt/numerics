import numpy as np
from matplotlib import pyplot as plt
m=10
x = np.linspace(0,1,m,endpoint = False)
print(x)
f = np.sin(2* 3.1415*x)
plt.plot(x,f)
plt.show()
