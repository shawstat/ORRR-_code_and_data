# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:25:28 2022

@author: shaw
"""


from functions_real import *

'Load the data'
Y = np.loadtxt('myUKdata.csv',skiprows=1,delimiter=',',usecols=(1,2,3,4))
X = np.loadtxt('myUKdata.csv',skiprows=1,delimiter=',',usecols=(5,6,7,8))
Y = Y.T
X = X.T

'Set the basic parameters'
ini_size = 50
test_size = 100
N = Y.shape[1]
p = Y.shape[0]
q = X.shape[0]
m=1           #The batch size we use in ORRR/ The number of data we add in each step for OSAA
k=3           #The index of which indicator will be plotted. 1——IPI, 2——MPI，3——RTI 4——CRI
name_list = ['Production of industry','Production of manufacturing','Retail trade','Car registrations']

'Set the initialization, training and test set'
Y_init = Y[:,:ini_size]
Y_test = Y[:,-test_size:]
Y_train = Y[:,ini_size:-test_size]
X_init = X[:,:ini_size]
X_test = X[:,-test_size:]
X_train = X[:,ini_size:-test_size]

result= np.zeros([2,4])

plt.figure()
plt.ylabel(name_list[k])
plt.xlabel('Time index')
plt.plot(np.arange(N-100,N),Y[k,N-100:],label = 'True')

for i in range(4):
    r = i+1
    
    'Estimation of ORRR'
    T = int(X_train.shape[1]/m)
    eta_c = np.ones(T)*(1e-5)
    eta_mu = np.ones(T)*(1e-4)
    t1 = time()
    A1,B1,mu1 = para_initi(X_init,Y_init,r,ini_size)
    est_A,est_B,est_mu = SGD_RRR(Y_train,X_train,A1,B1,mu1,m,eta_c,eta_mu,T)
    err_pre = error_pred(est_A,est_B,est_mu,Y_test,X_test)
    print('The ORRR for r=%d succeed in %f seconds'%(r,time()-t1))
    result[0,i]=err_pre[T-1]
    
    'Estimation of OSAAA'
    m_AA = 1
    m0_AA = ini_size
    X_train_AA = X_train[:,0::m_AA]
    Y_train_AA = Y_train[:,0::m_AA]
    X_train_AA.shape
    T_AA = X_train_AA.shape[1]-m0_AA
    A1_AA = np.zeros([p,r])
    B1_AA = np.zeros([p,r])
    mu1_AA = np.zeros(p)
    t2 = time()
    est_A_AA,est_B_AA,est_mu_AA = OSMM(Y_train_AA,X_train_AA,A1_AA,B1_AA,mu1_AA,m0_AA,1,T_AA)
    err_pre_AA = error_pred(est_A_AA,est_B_AA,est_mu_AA,Y_test,X_test)
    print('The OAAA for r=%d succeed in %f seconds'%(r,time()-t2))
    result[1,i]=err_pre_AA[T_AA-1]

    'Plot of the result'
    Y_pred = np.zeros([p,test_size])
    A = est_A[-1]
    B = est_B[-1]
    mu = est_mu[-1]
    # A = est_A_AA[-1]
    # B = est_B_AA[-1]
    # mu = est_mu_AA[-1]
    for i in range(test_size):
        Y_t = Y_test[:,i]
        X_t = X_test[:,i]
        Y_pred[:,i] = np.dot(np.dot(A,B.T),X_t)+mu


    plt.plot(np.arange(N-100,N),Y_pred[k,:],label = 'Predict,r=%d'%(r))
    
plt.legend()
#plt.savefig("Prod_Sale_%d.pdf"%(k))

'Save the result'
writer = pd.ExcelWriter('num_result_UKPS.xlsx')
pd.DataFrame(result).to_excel(writer, 'Sheet1',float_format='%.6f')
writer.save()
writer.close()