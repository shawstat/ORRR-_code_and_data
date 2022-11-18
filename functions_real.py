# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:10:29 2022

@author: shaw
"""
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
from time import time

def SGD_RRR(Y,X,A1,B1,mu1,m,eta_c,eta_mu,T):
    est_A = list()
    est_B = list()
    est_mu = list()
    A = A1
    B = B1
    mu = mu1
    p = Y.shape[0]
    
    for k in range(T):
        X_k = X[:,k*m:(k+1)*m]
        Y_k = Y[:,k*m:(k+1)*m]

        mu_mtx = np.repeat(mu,m).reshape([p,m])
        
        Diff  = mu_mtx+np.dot(np.dot(A,B.T),X_k)-Y_k
        
        balance = np.dot(A.T,A)-np.dot(B.T,B)
        
        Gmu = np.sum(Diff,axis=1)
        GA = np.dot(np.dot(Diff,X_k.T),B) +0.5*np.dot(A,balance)
        GB = np.dot(np.dot(X_k,Diff.T),A) +0.5*np.dot(B,-balance)
        
        mu = mu-eta_mu[k]*Gmu
        A = A-eta_c[k]*GA
        B = B-eta_c[k]*GB
        
        est_A.append(A)
        est_B.append(B)
        est_mu.append(mu)
    
    return(est_A,est_B,est_mu)

def mtx_sqrt_inv(A):
    U,S,V = np.linalg.svd(A)
    sig_sqrt_inv = np.sqrt(np.diag(1/S))
    A_sqrt_inv = np.dot(U,np.dot(sig_sqrt_inv,V))
    return(A_sqrt_inv)

def OSMM(Y,X,A1,B1,mu1,m0,m,T):
    est_A = list()
    est_B = list()
    est_mu = list()
    Ak = A1
    Bk = B1
    r = A1.shape[1]
    muk = mu1
    for k in range(T):
        nk = m0+k*m
        Pk = np.diag(np.ones(nk))-np.ones([nk,nk])/nk
        Yk = Y[:,:nk]
        Xk = X[:,:nk]
        Diff = Yk-np.dot(np.dot(Ak,Bk.T),Xk)
        muk = np.sum(Diff,axis=1)/nk  #The estimation of mu
        Mk = np.dot(Xk,Pk)
        Nk = np.dot(Yk,Pk)
        
        Rmm = np.dot(Mk,Mk.T)
        Rmn = np.dot(Mk,Nk.T)
        Rnm = Rmn.T
        Rnn = np.dot(Nk,Nk.T)
        
        TR = np.dot(mtx_sqrt_inv(Rmm),np.dot(Rmn,mtx_sqrt_inv(Rnn)))
        U,_,_ = np.linalg.svd(TR)
        Ur = U[:,:r]
        
        Ak = np.dot(Rnm,np.dot(mtx_sqrt_inv(Rmm),Ur))
        Bk = np.dot(mtx_sqrt_inv(Rmm),Ur)
        
        est_mu.append(muk)
        est_A.append(Ak)
        est_B.append(Bk)
    
    return(est_A,est_B,est_mu)

def error_para(est_A,est_B,est_mu,A0,B0,mu0):
    T = len(est_A)
    err_ab = np.zeros(T)
    err_mu = np.zeros(T)
    N = A0.shape[0]*B0.shape[0]
    p = A0.shape[0]
    C0 = np.dot(A0,B0.T)
    for i in range(T):
        E = C0-np.dot(est_A[i],est_B[i].T)
        err_ab[i] = (np.linalg.norm(E)**2)/N
        err_mu[i] = (np.linalg.norm(est_mu[i]-mu0)**2)/p
    return (err_ab,err_mu)

def error_pred(est_A,est_B,est_mu,Y_test,X_test):
    T = len(est_A)
    err_pre =np.zeros(T)
    m = X_test.shape[1]
    p = Y_test.shape[0]
    for i in range(T):
        A = est_A[i]
        B = est_B[i]
        mu = est_mu[i]
        Y_pred = np.dot(np.dot(A,B.T),X_test)+np.repeat(mu,m).reshape([p,m])
        err_pre[i] = (np.linalg.norm(Y_test-Y_pred)**2)/(m)
    
    return (err_pre)

def error_pre_online(est_A,est_B,est_mu,X,Y,m,ini_size,test_size):
    T = len(est_A)
    err_pre_online = np.zeros(T-test_size)
    p = Y.shape[0]
    for i in range(T-test_size):
        X_test = X[:,(i+1)*m:((i+1)*m+test_size)]
        Y_test = Y[:,(i+1)*m:((i+1)*m+test_size)]
        n = X_test.shape[1]
        A = est_A[i]
        B = est_B[i]
        mu = est_mu[i]
        Y_pred = np.dot(np.dot(A,B.T),X_test)+np.repeat(mu,n).reshape([p,n])
        err_pre_online[i] = (np.linalg.norm(Y_test-Y_pred)**2)/(p*n)
    return(err_pre_online)

def para_initi(X0,Y0,r,ini_size):

    mu_X = np.sum(X0,axis=1)/ini_size
    mu_Y = np.sum(Y0,axis=1)/ini_size
    
    'The least square solution'
    C1 = np.dot(np.linalg.inv(np.dot(X0,X0.T)),np.dot(X0,Y0.T)).T
    'The best r approximation'
    Uc,Sc,Vc = np.linalg.svd(C1)
    A1 = np.dot(Uc[:,:r],np.sqrt(np.diag(Sc[:r])))
    B1 = np.dot(np.sqrt(np.diag(Sc[:r])),Vc[:r,:]).T
    mu1 = mu_Y-np.dot(np.dot(A1,B1.T),mu_X)
    
    return(A1,B1,mu1)

