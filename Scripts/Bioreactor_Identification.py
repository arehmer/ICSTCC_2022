# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import sys
from pathlib import Path

toolbox_path = Path.cwd().parents[0]

sys.path.append(toolbox_path.as_posix())

import toolbox.models.NN as NN
from toolbox.optim import param_optim
from toolbox.optim import param_optim

from toolbox.optim.common import BestFitRate

# %% Data Preprocessing
################ Load Data ####################################################
train = loadmat('Benchmarks/Bioreactor/APRBS_Data_3')
train = train['data']
val = loadmat('Benchmarks/Bioreactor/APRBS_Data_1')
val = val['data']
test = loadmat('Benchmarks/Bioreactor/APRBS_Data_2')
test = test['data']


################ Subsample Data ###############################################
train = train[0::50,:]
val = val[0::50,:]
test = test[0::50,:]

################# Pick Training- Validation- and Test-Data ####################
train = [pd.DataFrame(data=train,columns=['u','y'])]
val = [pd.DataFrame(data=val,columns=['u','y'])]
test = [pd.DataFrame(data=test,columns=['u','y'])]

init_state = [np.zeros((2,1))]

data_train = {'data':train,'init_state':init_state}
data_val = {'data':val,'init_state':init_state}
data_test = {'data':test,'init_state':init_state}


# %% Identification

# Load inital linear state space model, identified with matlab on data_train
LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']

# %% Pick model by commenting out

''' Approach G-RNN '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = NN.G_RNN(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],y_label=['y'],
#                  dim_thetaA=1,dim_thetaB=0, dim_thetaC=0,fA_dim=0,fB_dim=0,
#                  fC_dim=0, initial_params=initial_params,name='G_RNN')

''' Approach S-RNN '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = NN.S_RNN(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],y_label=['y'],
#                  dim_thetaA=2,dim_thetaB=0,dim_thetaC=0,name='S_RNN')

''' Approach RBF '''
model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,u_label=['u'],y_label=['y'],
                    dim_theta=1, initial_params=None,name='RBF_network')

model.InitializeLocalModels(LSS['A'][0][0],LSS['B'][0][0],LSS['C'][0][0],
                            range_u = np.array([[0,0.7]]),
                            range_x = np.array([[0.004,0.231],[-0.248,0.0732]]))


''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

# Solver options
p_opts = {"expand":False}
s_opts = {"max_iter": 1000, "print_level":0,
'hessian_approximation': 'limited-memory'}

  
identification_results = param_optim.ModelTraining(model,data_train,
                                                    data_val,10,
                                                    p_opts=None,
                                                    s_opts=None)

# %% 
''' The output is a pandas dataframe which contains the results for each of
# the 10 initializations, specifically the loss on the validation data
# and the estimated parameters '''


# Pick best result
idx_opt = identification_results['loss_val'].idxmin()

model.Parameters = identification_results.loc[idx_opt,'params_val']

# Evaluate model on test data
_,predict = param_optim.parallel_mode(model,data_test)

# Plot the simulation result on test data to see how good the model performs
fig, ax = plt.subplots(1,1)
ax.plot(data_test['data'][0]['y'],label='True output')                                       # Plot True data
ax.plot(predict[0]['y'],label='Est. output')                                            # Plot Model Output
ax.legend()

