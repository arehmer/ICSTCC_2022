# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim
# from miscellaneous import *

from sklearn.preprocessing import MinMaxScaler


''' Data Preprocessing '''

################ Load Data ####################################################
SNLS80mV = pkl.load(open('Benchmarks/Silverbox/SNLS80mV.pkl','rb'))
Schroeder80mV = pkl.load(open('Benchmarks/Silverbox/Schroeder80mV.pkl','rb'))

################# Scale Data ##################################################
                                          
scaler = MinMaxScaler(feature_range=(-1,1))

# Validierungsdatensatz2 (Data_val) hat den größten Wertebereich, daher dieses Signal für Skalierung verwenden
SNLS80mV = pd.DataFrame(data = scaler.fit_transform(SNLS80mV),
                                  columns=SNLS80mV.columns)
Schroeder80mV = pd.DataFrame(data = scaler.transform(Schroeder80mV),
                                    columns=Schroeder80mV.columns)

################# Pick Training- Validation- and Test-Data ####################

train = SNLS80mV.iloc[40580:41270][['u','y']]-SNLS80mV.mean()       #SNLS80mV.iloc[40580:49270][['u','y']]-SNLS80mV.mean()
val = SNLS80mV.iloc[0:40580][['u','y']]-SNLS80mV.mean()
test = Schroeder80mV.iloc[10585:10585+1023][['u','y']]-Schroeder80mV.mean()

train_u = np.array(train[0:-1]['u']).reshape(1,-1,1)
train_y = np.array(train[1::]['y']).reshape(1,-1,1)

val_u = np.array(val[0:-1]['u']).reshape(1,-1,1)
val_y = np.array(val[1::]['y']).reshape(1,-1,1)

test_u = np.array(test[0:-1]['u']).reshape(1,-1,1)
test_y = np.array(test[1::]['y']).reshape(1,-1,1)


init_state = np.zeros((1,2,1))


# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u, 'y_val':val_y,'init_state_val': init_state,
        'u_test':val_u, 'y_test':val_y,'init_state_test': init_state}


''' Identification '''
# Load inital linear state space model
LSS=loadmat("./Benchmarks/Silverbox/Silverbox_LSS_s2")
LSS=LSS['LSS']


''' Approach Rehmer '''
initial_params = {'A': LSS['A'][0][0], 'B': LSS['B'][0][0], 'C': LSS['C'][0][0] }

model = NN.Rehmer_NN_LPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=0,dim_thetaC=0,
                  NN_A_dim=[[5,1]],NN_B_dim=[],NN_C_dim=[],
                  NN_A_act=[[1,1]],NN_B_act=[], NN_C_act=[], initial_params=None,
                  frozen_params = [], init_proc='random')

''' Approach Lachhab '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
#                           dim_thetaC=0, initial_params=None, frozen_params = [],
#                           init_proc='random')

''' RBF approach'''
initial_params = {'A0': LSS['A'][0][0], 'B0': LSS['B'][0][0], 'C0': LSS['C'][0][0],
                  'A1': LSS['A'][0][0], 'B1': LSS['B'][0][0], 'C1': LSS['C'][0][0]}
model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=2,
                      initial_params=initial_params)

''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

identification_results = param_optim.ModelTraining(model,data,5)

''' The output is a pandas dataframe which contains the results for each of
the 10 initializations, specifically the loss on the validation data
and the estimated parameters ''' 

# Pick the parameters from the first initialization (for example, in this case
# every model has a loss close to zero because the optimizer is really good
# and its 'only' a linear model which we identify)

# model.Parameters = identification_results.loc[0,'params']


# Maybe plot the simulation result on test data to see how good the model performs
# x_est,y_est = model.Simulation(init_state[0],test_u[0])

# y_est = np.array(y_est) 
 
 
# plt.plot(test_y[0],label='True output')                                        # Plot True data
# plt.plot(y_est,label='Est. output')                                            # Plot Model Output
# plt.plot(test_y[0]-y_est,label='Simulation Error')                             # Plot Error between model and true system (its almost zero)
# plt.legend()
# plt.show()


# Scatterplot of affine Parameters for visual inspection

# theta = np.array(theta) 
# plt.figure()
# plt.plot(thetaA[:,0],label='Theta_A1')    
# plt.plot(thetaA[:,1],label='Theta_A2')   
# plt.scatter(theta[:,0],theta[:,1])  
# plt.legend()
# plt.show()
# e2 = y[0]-y_est

# model.AffineStateSpaceMatrices([1,1])
