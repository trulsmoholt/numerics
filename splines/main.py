import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


xPoints = [-1,0,1]
yPoints = [0,1,0]
def bezier(x,i,j, t: float)->float:
    if j==0:
        return x[i]
    return (1-t)* bezier(x,i-1, j-1,t) + t * bezier(x, i, j-1,t)

xList = [] 
yList = []

for t in range(0,11):
    t = t/10
    xList.append(bezier(xPoints,2,2,t))
    yList.append(bezier(yPoints,2,2,t))

plt.plot(xList,yList,'r--',xPoints,yPoints,'g^')
plt.show()
print(xList)
