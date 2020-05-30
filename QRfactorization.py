import numpy as np

def houseHolderTransf(x):
    """ This function calculate the HouseHolder transformation matrix"""
    I = np.identity(len(x))
    b = np.zeros(len(x))
    b[0] = -1*np.sign(x[0])*np.linalg.norm(x)

    a_err = x - b
    a_err = a_err.reshape(len(x),1)
    P = I + np.dot(a_err,np.transpose(a_err))/(b[0]*a_err[0])
    return P

def QR(A):
    """This function calculate the factorization A = QR"""
    m,n = A.shape
    Q  = np.identity(m)
    R = A
    for i in range(n):
        P = houseHolderTransf(R[-m+i:,i])
        Qi  = np.identity(m)
        Qi[-len(P):,-len(P):] = P
        R = np.dot(Qi,R)
        Q = np.dot(Q,Qi)
    return Q,R

#Test
A = np.array([[-2,2,3],[4,5,4],[7,8,-9]])
Q,R = QR(A)

print 'Q:'
print Q.round(6)
print'R:'
print R.round(6)
#verify Q is orthogonal & t(Q)*A = R
aux = np.dot(Q,np.transpose(Q)) # this should ensamble the identity
print'Q*Q^t:'
print aux.round(6)
aux =  np.dot(np.transpose(Q),A) # this should be equal to R
print'Q^t*A:'
print aux.round(6)
