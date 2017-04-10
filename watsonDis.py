from scipy.special import hyp1f1
from scipy.special import gamma
import numpy as np

def get_MLmean(X):
    n,d = X.shape
    S = np.dot(X.T,X) #correlation matrix
    #S = S/n #normalization is unnecessary. Eigenvectors are invariant to scale
    D,V = np.linalg.eig(S)
    return V[:,0]

def get_MLkappa(mu,X):
    n,d = X.shape
    S = np.dot(X.T,X)
    S = S/n
    r = np.dot(mu.T,S).dot(mu)
#    print r
    a = 0.5
    c = float(d)/2
    # General aproximation
    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    # When r -> 1
    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    return BGG

def get_cp(Ndim, kappa):
    gammaValue = gamma(float(Ndim)/2)
    M = hyp1f1(0.5, float(Ndim)/2, kappa)   # Confluent hypergeometric function 1F1(a, b; x)
    cp = gammaValue / (np.power(2*np.pi, float(Ndim)/2)* M)
    return cp

def pdf (alpha, mu, kappa):
    # Watson pdf for a
    # mu: [mu0 mu1 mu...] p-1 dimesion angles in radians
    # kappa: Dispersion value
    # alpha: Vector of angles that we want to know the probability
####
    mu = np.array(mu).flatten()
    alpha = np.array(alpha).flatten()

    Ndim = mu.size #+ 1  ??
    cp = get_cp(Ndim, kappa)
#    print np.dot(mu.T, alpha)

    aux1 = np.dot(mu.T, alpha)
#    aux2 = 0
#    for i in range(mu.size):
#        aux2 = aux2 + mu[i]*alpha[i]
#    print alpha
    pdf = cp * np.exp(kappa * np.power(aux1,2))
    return pdf


def get_MLkappa(mu,X):
    n,d = X.shape
    S = np.dot(X.T,X)
    S = S/n

    r = np.dot(mu.T,S).dot(mu)
#    print r

    a = 0.5
    c = float(d)/2

    # General aproximation
    BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))

    # When r -> 1
    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    return BGG
