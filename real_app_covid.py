# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:15:02 2022

@author: shaw
"""


from functions_real import *

'Load the data for eight target states'
data_list = list()
state_list = ['Alabama', 'Florida', 'Georgia', 'Louisiana', 'Mississippi', 'North Carolina', 'South Carolina','Tennessee']
state_num = len(state_list)

for state in state_list:
    data_temp = np.loadtxt('coviddata\\%s.csv'%(state),skiprows=1,delimiter=',',usecols=(3))
    data_temp = data_temp[(-913):]
    data_list.append(data_temp)
    
result= np.zeros([2,8])
result_online_RR = list()
result_online_AA = list()


mydata = np.array(data_list)
p = mydata.shape[0]
q = p


'For each choice of rank r from 1 to 8, estimate the parameter with ORRR and OSAA'
for i in range(8):
    r=i+1     #The rank we use to estimate the parameter
    m=1       #The batch size we use in ORRR/ The number of data we add in each step for OSAA
    ini_size = 30
    test_size = 100
    lag = 1   # The lag number in the VAR model
    
    Y_init = mydata[:,lag:(ini_size+lag)]
    Y_test = mydata[:,-(test_size-lag):]
    Y_train = mydata[:,(ini_size+lag):-(test_size-lag)]
    
    X_init = mydata[:,0:ini_size]
    X_test = mydata[:,-(test_size):-lag]
    X_train = mydata[:,ini_size:-(test_size)]
    
    'Estimation of ORRR'
    T = int(X_train.shape[1]/m)
    eta_c = np.ones(T)*(1e-3)
    eta_mu = np.ones(T)*(4e-2)
    t1 = time()
    A1,B1,mu1 = para_initi(X_init,Y_init,r,ini_size)
    est_A,est_B,est_mu = SGD_RRR(Y_train,X_train,A1,B1,mu1,m,eta_c,eta_mu,T)
    err_pre = error_pred(est_A,est_B,est_mu,Y_test,X_test)
    print('The ORRR for r=%d succeed in %f seconds'%(r,time()-t1))
    
    'The real-time prediction error'
    X_online = mydata[:,ini_size:]
    Y_online = mydata[:,(ini_size+lag):]
    X_online = X_online[:,:-1]
    err_pre_online = error_pre_online(est_A,est_B,est_mu,X_online,Y_online,m,ini_size,10)
    
    result_online_RR.append(err_pre_online)
    result[0,i]=err_pre[T-1]
    
    
    'Estimation of OSAAA'
    m_AA = 1
    m0_AA = 30
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
    
    err_pre_online_AA = error_pre_online(est_A_AA,est_B_AA,est_mu_AA,X_online,Y_online,m,ini_size,10)
    result_online_AA.append(err_pre_online_AA)
    result[1,i]=err_pre_AA[T_AA-1]
    
writer = pd.ExcelWriter('num_result_covid.xlsx')
pd.DataFrame(result).to_excel(writer, 'Sheet1',float_format='%.6f')
writer.save()
writer.close()