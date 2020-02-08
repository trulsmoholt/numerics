import numpy as np
def forwardEuler(m):
    A = np.zeros([m,m])
    for i in range(0,m):
        if(i==m-1):
            A[i,i]=-1
            A[i,i-1]=0
            A[i,0]=1
        else:
            A[i,i]=-1
            A[i,i-1]=0
            A[i,i+1]=1
    
    return A