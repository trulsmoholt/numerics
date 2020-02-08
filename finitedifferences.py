import numpy as np
def forwardEuler(m):
    A = np.zeros([m,m])
    for i in range(0,m):
        if(i==m-1):
            A[i,i]=-1
            A[i,i-1]=1
            A[i,0]=0
        else:
            A[i,i]=-1
            A[i,i-1]=1
            A[i,i+1]=0
    
    return A

def centralDifference(m):
    A = np.zeros([m,m])
    for i in range(0,m):
        if(i==m-1):
            A[i,i]=0
            A[i,i-1]=-0.5
            A[i,0]=0.5
        else:
            A[i,i]=0
            A[i,i-1]=-0.5
            A[i,i+1]=0.5
    return A
