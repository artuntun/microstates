import numpy as np

A = np.array([[0.5,0.3,0.1],[0.5,0.4,0.1],[0,0,1]])
px_hot_cold = [np.array([0.2,0.4,0.4]),np.array([0.5,0.4,0.1])]
pi = np.array([0.8,0.2])
X = np.array([3,1,3,3,2,1])

X = [];Z = [];
def sampler(A,px_hot_cold,pi,N):
    zn = np.random.multinomial(1,pi,size=1)[0]
    state_dist = px_hot_cold[np.where(zn==1)[0][0]]
    xn = np.random.multinomial(1,state_dist,size=1)
    X.append(xn); Z.append(zn);
    for n in range(N):
        Ak = A[np.where(zn==1)[0][0],:][0][0]
        zn = np.random.multinomial(1,Ak,size=1)
        state_dist = px_hot_cold[np.where(zn==1)[0][0]]
        xn = np.random.multinomial(1,state_dist,size=1)
        X.append(xn); Z.append(zn)
    X = np.array(X).T; Z = np.array(Z).T
    return X,Z

X,Z = sampler(A,px_hot_cold,pi,N = 5)
Xnum = [np.where(x==1)[0][0] for x in X]
Znum = [np.where(z==1)[0][0] for z in Z]
