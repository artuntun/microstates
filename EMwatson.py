import watsonDis as ws
import numpy as np
import pandas as pd
import random 
def gen_unitvect():
    x1 = random.uniform(0,0.5)
    x2 = x1-0.3
    x3 = np.sqrt(1-np.power(x1,2)-np.power(x2,2))
    return np.array([x1,x2,x3])

def get_respons(pi,mu,kappa,X,numClusters):
    n,m = X.shape
    betas = np.zeros((n,numClusters))
    """Calculate the probability of each point to belong to each cluster"""
    for i in range(0,n):
        denom = 0
        for l in range(0,numClusters):
            denom = denom + pi[l]*ws.pdf(X[i,:],mu[l],kappa[l])
        for j in range(0,numClusters):
            numera = pi[j]*ws.pdf(X[i,:],mu[j],kappa[j])
            betas[i,j] = numera/denom
    return betas

def get_kappa(Sj,numClusters,newMu):
    o = np.dot(newMu,Sj)
    r = np.dot(o,newMu)
    a = 1./2
    c = float(numClusters)/2
    BGG = (c - a)/(1-r) + 1 - a + (a - 1)*(a-c-1)*(1-r)/(c-a)
    #BGG = (c*r -a)/(r*(1-r)) + r/(2*c*(1-r))
    return BGG 

def get_mus(X,betas,kappa):
    n,m = X.shape
    Xi = np.empty_like (X)
    Xi[:,:] = X
    for i in range(0,n):
        Xi[i,:]=Xi[i,:]*betas[i]

    S = np.dot(Xi.T,X)
    Nk = np.sum(betas)  #expected number of points to belong to the cluster
    S = S/Nk
    D,V = np.linalg.eig(S)
    if kappa >= 0:
        newMu = V[:,0]
    else:
        newMu = V[:,1]
    return newMu,S

#get data from 3 Watson distributions
path = "clust1.csv"
path2 = "clust2.csv"
path3 = "clust3.csv"
X1 = np.array(pd.read_csv(path, sep = ","))
X2 = np.array(pd.read_csv(path2, sep = ","))
X3 = np.array(pd.read_csv(path3, sep = ","))
X = np.concatenate((X1,X2,X3[0:1400,:]),axis=0)
n,m = X.shape

#initialization and memory allocation
numClusters = 3
pi = np.zeros(numClusters)
pi = pi+(1./numClusters)
mu = np.zeros((numClusters,m))
kappa = np.zeros(numClusters)
for i in range(0,numClusters):
    mu[i,:] = gen_unitvect()
    kappa[i] = random.uniform(10,20)
muList = [mu]
kappaList = [kappa]
piList = [pi]

maxit = 100
it = 0
tol = 1000
#EM algorithm starts
while tol >10e-5 and it < maxit:
    mu = muList[it]
    kappa = kappaList[it]
    pi = piList[it]
    newMu = np.zeros((numClusters,m))
    newKappa = np.zeros((numClusters))
    newPi = np.zeros((numClusters))
    # E-step
    betas = get_respons(pi,mu,kappa,X,numClusters)
    #M-step
    for j in range(0,numClusters):
        #Calculate mu
        newMu[j,:],Sj = get_mus(X,betas[:,j],kappa[j])
        #Calucalte kappa
        newKappa[j] = get_kappa(Sj,numClusters,newMu[j,:])
        #Calculate pies
        Nk = np.sum(betas[:,j])  #expected number of points to belong to the cluster
        newPi[j] = Nk/n
    
    muList.append(newMu)
    kappaList.append(newKappa)
    piList.append(newPi)
    #meter loglikelihood
    tol = (np.linalg.norm(muList[it]-newMu)+np.linalg.norm(piList[it]-newPi)+
            np.linalg.norm(kappaList[it]-newKappa))
    it +=1

import IPython
IPython.embed()
