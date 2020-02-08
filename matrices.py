import numpy as np
Q = np.array([[1,1,1],[2,1,0],[0,1,1]])
def tensor(v1,v2,A):
    t = np.transpose(v1) @ A @ v2
    return t

omega = 1.0
A = np.eye(3)
A[0,1]=omega
print("Q=","\n",Q)
print("A=","\n",A)

v1 = np.array([1,1,1])
v2 = np.array([2,3,4])
T=np.transpose(Q) @ A @ Q
print("T=","\n",T)

L = np.linalg.inv(Q) @ A @ Q

print("tensor product of ",v1," and ",v2," with A is " ,tensor(v1,v2,A))
v1 = np.linalg.inv(Q) @ v1
v2 = np.linalg.inv(Q) @ v2
print("tensor product of ",v1," and ",v2," with A in new basis is " ,tensor(v1,v2,T))


print("L=","\n",L)
print("v1=","\n",v1)