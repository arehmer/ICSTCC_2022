#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

import os

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import pickle as pkl
import copy

# from .DiscreteBoundedPSO import DiscreteBoundedPSO
from .common import OptimValues_to_dict
from .common import BestFitRate,AIC

from ..miscellaneous.PreProcessing import arrange_ARX_data

# Import sphere function as objective function
#from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
# import pyswarms.backend as P
# from pyswarms.backend.topology import Star
# from pyswarms.discrete.binary import BinaryPSO

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


# from miscellaneous import *


def ControlInput(ref_trajectories,opti_vars,k):
    """
    Übersetzt durch Maschinenparameter parametrierte
    Führungsgrößenverläufe in optimierbare control inputs
    """
    
    control = []
            
    for key in ref_trajectories.keys():
        control.append(ref_trajectories[key](opti_vars,k))
    
    control = cs.vcat(control)

    return control   
    
def CreateOptimVariables(opti, Parameters):
    """
    Defines all parameters, which are part of the optimization problem, as 
    opti variables with appropriate dimensions
    """
    
    # Create empty dictionary
    opti_vars = {}
    
    for param in Parameters.keys():
        dim0 = Parameters[param].shape[0]
        dim1 = Parameters[param].shape[1]
        
        opti_vars[param] = opti.variable(dim0,dim1)
    
    return opti_vars

def ModelTraining(model,data_train,data_val,initializations=10, BFR=False, 
                  p_opts=None, s_opts=None,mode='parallel'):
    
   
    results = [] 
    
    for i in range(0,initializations):
        
        res = TrainingProcedure(model, data_train,data_val, p_opts, s_opts, mode)
               
        # save parameters and performance in list
        results.append(res)
           
    results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
                        'model','params_train','params_val'])
    return results

def TrainingProcedure(model, data_train, data_val, p_opts, s_opts, mode):
    
    # initialize model to make sure given initial parameters are assigned
    model.ParameterInitialization()
    
    # Estimate Parameters on training data
    params_train,params_val,loss_train,loss_val = \
        ModelParameterEstimation(model,data_train,data_val,p_opts,s_opts,mode)
    
    # save parameters and performance in list
    result = [loss_train,loss_val,model.name,params_train,params_val]
    
    return result

def ParallelModelTraining(model,data_train,data_val,initializations=10,
                          BFR=False, p_opts=None, s_opts=None,mode='parallel',
                          n_pool=5):
    
     
    data_train = [copy.deepcopy(data_train) for i in range(0,initializations)]
    data_val = [copy.deepcopy(data_val) for i in range(0,initializations)]
    model = [copy.deepcopy(model) for i in range(0,initializations)]
    p_opts = [copy.deepcopy(p_opts) for i in range(0,initializations)]
    s_opts = [copy.deepcopy(s_opts) for i in range(0,initializations)]
    mode = [copy.deepcopy(mode) for i in range(0,initializations)]
    
    pool = multiprocessing.Pool(n_pool)
    results = pool.starmap(TrainingProcedure, zip(model, data_train, data_val, p_opts, s_opts, mode))        
    results = pd.DataFrame(data = results, columns = ['loss_train','loss_val',
                        'model','params_train','params_val'])
    
    pool.close() 
    pool.join()      
    
    return results 

# 

def ModelParameterEstimation(model,data_train,data_val,p_opts=None,
                             s_opts=None,mode='parallel'):
    """
    

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    values : dict
        dictionary with either the optimal parameters or if the solver did not
        converge the last parameter estimate

    """
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create dictionary of all non-frozen parameters to create Opti Variables of 
    OptiParameters = copy.deepcopy(model.Parameters)
    
    for frozen_param in model.frozen_params:
        OptiParameters.pop(frozen_param)
        
    
    params_opti = CreateOptimVariables(opti, OptiParameters)   
    
    # Evaluate on model on data
    
    if mode == 'parallel':
        
        loss_train,_ = parallel_mode(model,data_train,params_opti)
        loss_val,_ = parallel_mode(model,data_val,params_opti)
        
    elif mode == 'static':
        loss_train,_,_ = static_mode(model,u_train,y_ref_train,params_opti)   
        loss_val,_,_ = static_mode(model,u_val,y_ref_val,params_opti) 
                
    elif mode == 'series':      
        loss_train,_ = series_parallel_mode(model,data_train,params_opti)
        loss_val,_ = series_parallel_mode(model,data_val,params_opti)
    
    loss_val = cs.Function('loss_val',[*list(params_opti.values())],
                         [loss_val],list(params_opti.keys()),['F'])
     
    opti.minimize(loss_train)

    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 1000, "print_level":1}
        
    # Create Solver
    opti.solver("ipopt",p_opts, s_opts)
        
    class intermediate_results():
        def __init__(self):
            self.F_val = np.inf
            self.params_val = {}
            
        def callback(self,i):
            params_val_new = OptimValues_to_dict(params_opti,opti.debug)
            
            F_val_new = loss_val(*list(params_val_new.values()))

            if F_val_new < self.F_val:
                self.F_val = F_val_new
                self.params_val = params_val_new
                print('Validation loss: ' + str(self.F_val))
    
    val_results = intermediate_results()

    # Callback
    opti.callback(val_results.callback)
    
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])

    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    params = OptimValues_to_dict(params_opti,sol)
    F_train = sol.value(opti.f)        

    params_val = val_results.params_val
    F_val = val_results.F_val
    
            
    return params,params_val,float(F_train),float(F_val)




def series_parallel_mode(model,data,params=None):
  
    loss = 0
    
    x = []
    
    prediction = []

    # if None is not None:
        
    #     print('This is not implemented properly!!!')
        
    #     for i in range(0,len(u)):
    #         x_batch = []
    #         y_batch = []
            
    #         # One-Step prediction
    #         for k in range(u[i].shape[0]-1):
                
                
                
    #             x_new,y_new = model.OneStepPrediction(x_ref[i][k,:],u[i,k,:],
    #                                                   params)
    #             x_batch.append(x_new)
    #             y_batch.append(y_new)
                
    #             loss = loss + cs.sumsqr(y_ref[i][k,:]-y_new) + \
    #                 cs.sumsqr(x_ref[i,k+1,:]-x_new) 
            
    #         x.append(x_batch)
    #         y.append(y_batch)
        
    #     return loss,x,y 
    
    # else:
    for i in range(0,len(data['data'])):
        
        io_data = data['data'][i]
        x0 = data['init_state'][i]
        switch = data['switch'][i]
        
        y_est = []
        # One-Step prediction
        for k in range(0,io_data.shape[0]-1):
            
            uk = io_data.iloc[k][model.u_label].values.reshape((-1,1))
            yk = io_data.iloc[k][model.y_label].values.reshape((-1,1))
            
            ykplus = io_data.iloc[k+1][model.y_label].values.reshape((-1,1))
            
            # predict x1 and y1 from x0 and u0
            y_new = model.OneStepPrediction(yk,uk,params)
            
            loss = loss + cs.sumsqr(ykplus-y_new)        
            
            y_est.append(y_new.T)
        
        y_est = cs.vcat(y_est)
        
        if params is None:
            y_est = np.array(y_est)
            
            df = pd.DataFrame(data=y_est, columns=model.y_label,
                              index=io_data.index[1::])
            
            prediction.append(df)
        else:
            prediction = None
        
    return loss,prediction


def EstimateNonlinearStateSequenceKF(model,data,lam):
    """
    

    Parameters
    ----------
    model : LinearSSM
        Linear state space model.
    data : dict
        dictionary containing input and output data as numpy arrays with keys
        'u_train' and 'y_train'
    lam : float
        trade-off parameter between fit to data and linear model fit, needs to
        be positive

    Returns
    -------
    x_LS: array-like
        numpy-array containing the estimated nonlinear state sequence

    """
    for i in range(0,len(data['data'])):
        
        
        if i>1:
            print('This program is designed for only one batch of data')
            
        
        io_data = data['data'][i]
        x0 = data['init_state'][i]
        
        N = io_data.shape[0]
        num_states = model.dim_x
        # switch = data['switch'][i]
        
        # Create Instance of the Optimization Problem
        opti = cs.Opti()        

        # Create decision variables for states
        x_LS = opti.variable(N,num_states) # x0,...,xN-1
        x_LS[0,:] = x0.T
        
        loss = 0
        
        # One-Step prediction
        for k in range(0,N-1):
            
            uk = io_data.iloc[k][model.u_label].values
            yk_plus = io_data.iloc[k+1][model.y_label].values
            
            # predict x1 and y1 from x0 and u0
            x_new,y_new = model.OneStepPrediction(x_LS[[k],:],uk)

            loss = loss + cs.sumsqr(yk_plus - y_new)  + \
                + lam * cs.sumsqr(x_LS[[k+1],:].T - x_new)   

        
    opti.minimize(loss)    
    opti.solver("ipopt")
    
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
       
      
    x_LS = np.array(sol.value(x_LS)).reshape(N,num_states)
    x_LS = pd.DataFrame(data=x_LS,columns=['x_LS'],index=io_data.index)
      
    return x_LS

def EstimateNonlinearStateSequenceEKF(model,data,lam):
    """
    

    Parameters
    ----------
    model : LinearSSM
        Linear state space model.
    data : dict
        dictionary containing input and output data as numpy arrays with keys
        'u_train' and 'y_train'
    lam : float
        trade-off parameter between fit to data and linear model fit, needs to
        be positive

    Returns
    -------
    x_LS: array-like
        numpy-array containing the estimated nonlinear state sequence

    """
    for i in range(0,len(data['data'])):
        
        
        if i>1:
            print('This program is designed for only one batch of data')
            
        
        io_data = data['data'][i]
        x0 = data['init_state'][i]
        
        N = io_data.shape[0]
        num_states = model.dim_x
        # switch = data['switch'][i]
        
        # Create Instance of the Optimization Problem
        opti = cs.Opti()        

        # Create decision variables for states
        x_LS = opti.variable(N,num_states) # x0,...,xN-1
        x_LS[0,:] = x0.T
        
        # x_LS = []
        # x_LS.append(x0.T)
        
        loss = 0
        

        
        # One-Step prediction
        for k in range(0,N-1):

            # Create Instance of the Optimization Problem
            # opti = cs.Opti()   
            
            # New state is the target of optimization
            # x_new = opti.variable(1,num_states)
            
        
            
            uk = io_data.iloc[k][model.u_label].values
            
            # do a one step prediction given the model
            # pred = model.OneStepPrediction(x0,uk)
            pred = model.OneStepPrediction(x_LS[k,:],uk)
            
            A = pred['dfdx']
            B = pred['dfdu']
            C = pred['dgdx']
                     
            
            # dx = x_new.T-x0
            dx = x_LS[k+1,:].T - x_LS[k,:].T
            
            x_pred = pred['x_new'] # cs.mtimes(A,dx) + cs.mtimes(B,uk) + pred['x_new']          # estimate of x_k+1
            y_pred = cs.mtimes(C,dx) + pred['y_old']                            # estimate of y_k+1
            
            
            # x_new = pred['x_new']
            # y_new = pred['y_new']
            
            yk_plus = io_data.iloc[k+1][model.y_label].values
            
            # loss = cs.sumsqr(yk_plus - y_pred)  + \
            #     + lam * cs.sumsqr(x_new.T - x_pred)   
                
                
            loss = loss + cs.sumsqr(yk_plus - y_pred)  + \
                + lam * cs.sumsqr(x_LS[k+1,:].T - x_pred)  
                
            # opti.minimize(loss) 
            
            # opti.solver('ipopt')
            
            # sol = opti.solve()
            
            # x_new = np.array(sol.value(x_new)).reshape(1,num_states) 
            
            # x_LS.append(x_new)
            
            # x0 = x_new
            
    opti.minimize(loss)    
    opti.solver("ipopt")
    
    try:
        x_init = io_data['x_LS'] 

        for i in range(0,x_LS.shape[0]):
            opti.set_initial(x_LS[i],x_init.iloc[i]) 
              
    except KeyError:
        pass

    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        

      
    x_LS = np.array(sol.value(x_LS)).reshape(N,num_states)
    # x_LS = np.array(x_LS).reshape(N,num_states)
    x_LS = pd.DataFrame(data=x_LS,columns=['x_LS'],index=io_data.index)
      
    return x_LS

def EKF_Filter(model,io_data,w,v):
    
    state_est = pd.DataFrame(data=[],index=io_data.index,
                             columns=['x_prio','x_post','Pf_prio','Pf_post',
                                      'F','B'])
    
    # Initialize Kalman-Filter
    try:
        state_est['x_prio'].iloc[0] = io_data['x_ref'].iloc[0]
        state_est['x_post'].iloc[0] = io_data['x_ref'].iloc[0]
    except KeyError:
        state_est['x_prio'].iloc[0] = np.zeros((model.dim_x,1))
        state_est['x_post'].iloc[0] = np.zeros((model.dim_x,1))


    state_est['Pf_prio'].loc[0] = np.zeros((model.dim_x,model.dim_x))
    state_est['Pf_post'].loc[0] = np.zeros((model.dim_x,model.dim_x))

    # Noise covariance matrices 
    Q = np.eye(model.dim_x) * w
    R = np.eye(model.dim_y) * v
    
    for k in io_data.index[1::]:
        
        x_post = state_est['x_post'].iloc[k-1]
        x_prio = state_est['x_prio'].iloc[k-1]
        
        # Linearization around x_post of last time step for state transition
        uk = io_data.iloc[k-1][model.u_label].values
        pred = model.OneStepPrediction(x0=x_post,u0=uk)
        
        F = np.array(pred['dfdx'])
        B = np.array(pred['dfdu'])
    
        # Propagate state
        x_prio = np.array(pred['x_new'])
    
        # Linearization around x_prio of this time step for output equation        
        pred = model.OneStepPrediction(x0=x_prio,u0=uk)
        
        H = np.array(pred['dgdx'])
        y_prio = np.array(pred['y_old'])
        
        # Error covariance propagation
        P_old = state_est['Pf_post'].loc[k-1]
        P_new = F.dot(P_old).dot(F.T) + Q
        
        
        # Kalman gain matrix
        g = np.linalg.inv(H.dot(P_new).dot(H.T)+R)
        G = P_new.dot(H.T).dot(g)
        
        # State estimate update
        y = io_data.loc[k][model.y_label].values.reshape((-1,1))
        x_post = x_prio + G.dot(y-y_prio)
        
        # Error covariance update
        state_est['Pf_prio'].loc[k] = P_new
        state_est['Pf_post'].loc[k] = (np.eye(model.dim_x) - G.dot(H)).dot(P_new)
            
        
        # Save estimated states
        state_est['x_post'].loc[k] = x_post
        state_est['x_prio'].loc[k] = x_prio
        
        # Save linearization
        state_est['F'].loc[k] = F
        state_est['B'].loc[k] = B
        
    return state_est
        
def RTS_Smoother(model,io_data,w,v):
    
    # Apply EKF for forward filtering
    forward_est = EKF_Filter(model,io_data,w,v)

    smooth_est = pd.DataFrame(data=[],index=io_data.index,
                             columns=['x_prio','x_post','Pf_prio','Pf_post',
                                      'F','B','P','x'])
    # Initialize smoother
    idx = io_data.index[-1]
    
    smooth_est['P'].loc[idx] = forward_est['Pf_post'].loc[idx]
    smooth_est['x'].loc[idx] = forward_est['x_post'].loc[idx]
    
    
    for k in reversed(io_data.index[0:-1]):
        
        Pf_post = forward_est['Pf_post'].loc[k]
        Pf_prio = forward_est['Pf_prio'].loc[k+1]
        F = forward_est['F'].loc[k+1]
        
        P = smooth_est['P'].loc[k+1]
        x = smooth_est['x'].loc[k+1]
        
        A = Pf_post.dot(F.T).dot(np.linalg.inv(Pf_prio))
        
        P = Pf_post - A.dot(Pf_prio-P).dot(A.T)
        
        x =  forward_est['x_post'].loc[k] + A.dot(x-forward_est['x_prio'].loc[k+1])
        
        smooth_est['P'].loc[k] = P
        smooth_est['x'].loc[k] = x
    
    return smooth_est, forward_est
        

def ARXParameterEstimation(model,data,p_opts=None,s_opts=None, mode='parallel'):
    """
    

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    values : dict
        dictionary with either the optimal parameters or if the solver did not
        converge the last parameter estimate

    """
   
    u = data['u_train']
    y_in = data['y_in']
    y_ref = data['y_train']
      
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create dictionary of all non-frozen parameters to create Opti Variables of 
    OptiParameters = model.Parameters.copy()
    
    for frozen_param in model.frozen_params:
        OptiParameters.pop(frozen_param)
        
    
    params_opti = CreateOptimVariables(opti, OptiParameters)
    
    e = 0
    
    # Training in series parallel configuration        
    # Loop over all batches 
    for i in range(0,u.shape[0]):  
        
        # One-Step prediction
        for k in range(u[i,:,:].shape[0]-1):  
            # print(k)
            
            Y = y_in[i,k,:].reshape((model.shifts,model.dim_y)).T
            U = u[i,k,:].reshape((model.shifts,model.dim_u)).T
            
            y_new = model.OneStepPrediction(Y,U,params_opti)
        
            # Calculate one step prediction error
            e = e + cs.sumsqr(y_ref[i,k,:]-y_new)

    opti.minimize(e)
        
    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 3000, "print_level":1}

    # Create Solver
    opti.solver("ipopt",p_opts, s_opts)
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])
    
    
    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    values = OptimValues_to_dict(params_opti,sol)
    
    return values

def ARXOrderSelection(model,u,y,order=[i for i in range(1,20)],p_opts=None,
                      s_opts=None):

    results = []
    
    for o in order:
        print(o)
        # Arange data according to model order
       
        y_ref, y_shift, u_shift = arrange_ARX_data(u=u,y=y,shifts=o)

        y_ref = y_ref.reshape(1,-1,model.dim_y)
        y_shift = y_shift.reshape(1,-1,o*model.dim_u)
        u_shift = u_shift.reshape(1,-1,o*model.dim_u)


        data = {'u_train':u_shift,'y_train':y_ref, 'y_in':y_shift}
        
        
        setattr(model,'shifts',o)
        model.Initialize()
        
        params = ARXParameterEstimation(model,data)
        
        model.Parameters = params
        
        # Evaluate estimated model on first batch of training data in parallel mode        
        _,y_NARX = model.Simulation(y_shift[0,[0],:],u_shift[0])
        
        y_NARX = np.array(y_NARX)
        
        
        # Calculate AIC
        aic = AIC(y_ref[0],y_NARX,model.num_params,p=2)

        # save results in list
        results.append([o,model.num_params,aic,params])

   
    results = pd.DataFrame(data = results, columns = ['order','num_params',
                                                      'aic','params'])
    
    
    return results
    
    
# def parallel_mode(model,data,params=None):
      
#     loss = 0

    
#     simulation = []
    
#     # Loop over all batches 
#     for i in range(0,len(data['data'])):
        
#         io_data = data['data'][i]
#         x0 = data['init_state'][i]
#         try:
#             switch = data['switch'][i]
#             switch = [io_data.index.get_loc(s) for s in switch]
#             print('Kontrolliere ob diese Zeile gewünschte Indizes zurückgibt!!!')               
#         except KeyError:
#             switch = None
        
#         kwargs = {'switching_instances':switch}
        
#         y_ref = io_data.iloc[1::][model.y_label].values                         # y_1,...,y_N
        
#         u = io_data.iloc[0:-1][model.u_label].values                            # u_0,...,u_N-1

#         # Simulate Model
#         pred = model.Simulation(x0,u,params,**kwargs)
        
#         if isinstance(pred, tuple):           
#             x_est= pred[0]
#             y_est= pred[1]
#         else:
#             y_est = pred                                                        # y_1,...,y_N
            
#         # Calculate simulation error            
#         # Check for the case, where only last value is available
        
#         if y_ref.shape[0]==1:           # MUST BE UPDATED TO WORK WITH QUALITY DATA

#             y_est=y_est[-1,:]
#             e= y_ref - y_est
#             loss = loss + cs.sumsqr(e[-1])
    
#         else :
#             e = y_ref - y_est
#             loss = loss + cs.sumsqr(e)
            
#         if params is None:
#             y_est = np.array(y_est)
            
#             df = pd.DataFrame(data=y_est, columns=model.y_label,
#                               index=io_data.index)
            
#             simulation.append(df)
#         else:
#             simulation = None
            
#     return loss,simulation

def parallel_mode(model,data,params=None):
      
    loss = 0

    
    simulation = []
    
    # Loop over all batches 
    for i in range(0,len(data['data'])):
        
        io_data = data['data'][i]
        x0 = data['init_state'][i]
        try:
            switch = data['switch'][i]
            # switch = [io_data.index.get_loc(s) for s in switch]
            kwargs = {'switching_instances':switch}
                        
            # print('Kontrolliere ob diese Zeile gewünschte Indizes zurückgibt!!!')               
        except KeyError:
            switch = None
            kwargs = {'switching_instances':switch}
        
        u = io_data.iloc[0:-1][model.u_label]
        # u = io_data[model.u_label]

        
        # Simulate Model        
        pred = model.Simulation(x0,u,params,**kwargs)
        
        
        y_ref = io_data[model.y_label]
        
        
        if isinstance(pred, tuple):           
            x_est= pred[0]
            y_est= pred[1]
        else:
            y_est = pred
            
        # Calculate simulation error            
        # Check for the case, where only last value is available
        
        if np.all(np.isnan(y_ref.iloc[0:])):
            
            y_ref = y_ref.iloc[0].values
            y_est=y_est[-1,:]
            e= y_ref - y_est
            loss = loss + cs.sumsqr(e)
            
            idx = [i]
    
        else :
            
            y_ref = y_ref.iloc[1:1+y_est.shape[0]]                                # first observation cannot be predicted
            
            e = y_ref.values - y_est
            loss = loss + cs.sumsqr(e)
            
            idx = y_ref.index
        
        if params is None:
            y_est = np.array(y_est)
            
            df = pd.DataFrame(data=y_est, columns=model.y_label,
                              index=idx)
            
            simulation.append(df)
        else:
            simulation = None
            
    return loss,simulation
def static_mode(model,u,y_ref,params=None):
    
    loss = 0
    y = []
    e = []
    
                    
    # Loop over all batches 
    for i in range(0,len(u)): 

        # One-Step prediction
        for k in range(u[i].shape[0]):  
            # print(k)
            y_new = model.OneStepPrediction(u[i][k,:],params)
            # print(y_new)
            y.append(y_new)
            e.append(y_ref[i][k,:]-y_new)
            # Calculate one step prediction error
            loss = loss + cs.sumsqr(e[-1]) 
            
        break
    
    return loss,e,y

def series_parallel_mode(model,data,params=None):
  
    loss = 0
    
    x = []
    
    prediction = []


    for i in range(0,len(data['data'])):
        
        io_data = data['data'][i]
        x0 = data['init_state'][i]
       
        try:
            switch = data['switch'][i]
            switch = [io_data.index.get_loc(s) for s in switch]
            print('Kontrolliere ob diese Zeile gewünschte Indizes zurückgibt!!!')               
        except KeyError:
            switch = None
        
        if 'x_ref' in io_data.keys():
            
            x_est = []
            y_est = []
            
            # One-Step prediction
            for k in range(0,io_data.shape[0]-1):
                
                uk = io_data.iloc[k][model.u_label].values.reshape((-1,1))
                xk = io_data.iloc[k][['x_ref']].values.reshape((-1,1))
                
                xkplus = io_data.iloc[k+1][['x_ref']].values.reshape((-1,1))        # TO DO: FÜR dim_x>1 anpassen
                ykplus = io_data.iloc[k+1][model.y_label].values.reshape((-1,1))
                
                # predict x1 and y1 from x0 and u0
                pred = model.OneStepPrediction(xk,uk,params)
                x_new = pred['x_new']
                y_new = pred['y_new']
                
                loss = loss + cs.sumsqr(ykplus-y_new) + cs.sumsqr(xkplus-x_new)        
                
                x_est.append(x_new.T)
                y_est.append(y_new.T)
            
            x_est = cs.vcat(x_est) 
            y_est = cs.vcat(y_est)            
        
            if params is None:
                x_est = np.array(x_est)
                y_est = np.array(y_est)
                
                df = pd.DataFrame(data=np.hstack([y_est,x_est]), 
                                  columns=[*model.y_label,'x_est'],
                                  index=io_data.index[1::])
                
                prediction.append(df)
            else:
                prediction = None
        else:
        
            y_est = []
        
            # One-Step prediction
            for k in range(0,io_data.shape[0]-1):
                
                uk = io_data.iloc[k][model.u_label].values.reshape((-1,1))
                yk = io_data.iloc[k][model.y_label].values.reshape((-1,1))
                
                ykplus = io_data.iloc[k+1][model.y_label].values.reshape((-1,1))
                
                # predict x1 and y1 from x0 and u0
                pred = model.OneStepPrediction(yk,uk,params)
                y_new = pred['y_new']
                
                loss = loss + cs.sumsqr(ykplus-y_new)        
                
                y_est.append(y_new.T)
            
            y_est = cs.vcat(y_est)
            
            if params is None:
                y_est = np.array(y_est)
                
                df = pd.DataFrame(data=y_est, columns=model.y_label,
                                  index=io_data.index[1::])
                
                prediction.append(df)
            else:
                prediction = None
        
    return loss,prediction

# def HyperParameterPSO(model,data,param_bounds,n_particles,options,
#                       initializations=10,p_opts=None,s_opts=None):
#     """
#     Binary PSO for optimization of Hyper Parameters such as number of layers, 
#     number of neurons in hidden layer, dimension of state, etc

#     Parameters
#     ----------
#     model : model
#         A model whose hyperparameters to be optimized are attributes of this
#         object and whose model equations are implemented as a casadi function.
#     data : dict
#         A dictionary with training and validation data, see ModelTraining()
#         for more information
#     param_bounds : dict
#         A dictionary with structure {'name_of_attribute': [lower_bound,upper_bound]}
#     n_particles : int
#         Number of particles to use
#     options : dict
#         options for the PSO, see documentation of toolbox.
#     initializations : int, optional
#         Number of times the nonlinear optimization problem is solved for 
#         each particle. The default is 10.
#     p_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.
#     s_opts : dict, optional
#         options to give to the optimizer, see Casadi documentation. The 
#         default is None.

#     Returns
#     -------
#     hist, Pandas Dataframe
#         Returns Pandas dataframe with the loss associated with each particle 
#         in the first column and the corresponding hyperparameters in the 
#         second column

#     """
    
    
#     # Formulate Particle Swarm Optimization Problem
#     dimensions_discrete = len(param_bounds.keys())
#     lb = []
#     ub = []
    
#     for param in param_bounds.keys():
        
#         lb.append(param_bounds[param][0])
#         ub.append(param_bounds[param][1])
    
#     bounds= (lb,ub)
    
#     # Define PSO Problem
#     PSO_problem = DiscreteBoundedPSO(n_particles, dimensions_discrete, 
#                                      options, bounds)

#     # Make a directory and file for intermediate results 
#     os.mkdir(model.name)

#     for key in param_bounds.keys():
#         param_bounds[key] = np.arange(param_bounds[key][0],
#                                       param_bounds[key][1]+1,
#                                       dtype = int)
    
#     index = pd.MultiIndex.from_product(param_bounds.values(),
#                                        names=param_bounds.keys())
    
#     hist = pd.DataFrame(index = index, columns=['cost','model_params'])    
    
#     pkl.dump(hist, open(model.name +'/' + 'HyperParamPSO_hist.pkl','wb'))
    
#     # Define arguments to be passed to vost function
#     cost_func_kwargs = {'model': model,
#                         'param_bounds': param_bounds,
#                         'n_particles': n_particles,
#                         'dimensions_discrete': dimensions_discrete,
#                         'initializations':initializations,
#                         'p_opts': p_opts,
#                         's_opts': s_opts}
    
#     # Create Cost function
#     def PSO_cost_function(swarm_position,**kwargs):
        
#         # Load training history to avoid calculating stuff muliple times
#         hist = pkl.load(open(model.name +'/' + 'HyperParamPSO_hist.pkl',
#                                  'rb'))
            
#         # except:
            
#         #     os.mkdir(model.name)
            
#         #     # If history of training data does not exist, create empty pandas
#         #     # dataframe
#         #     for key in param_bounds.keys():
#         #         param_bounds[key] = np.arange(param_bounds[key][0],
#         #                                       param_bounds[key][1]+1,
#         #                                       dtype = int)
            
#         #     index = pd.MultiIndex.from_product(param_bounds.values(),
#         #                                        names=param_bounds.keys())
            
#         #     hist = pd.DataFrame(index = index, columns=['cost','model_params'])
        
#         # Initialize empty array for costs
#         cost = np.zeros((n_particles,1))
    
#         for particle in range(0,n_particles):
            
#             # Check if results for particle already exist in hist
#             idx = tuple(swarm_position[particle].tolist())
            
#             if (math.isnan(hist.loc[idx,'cost']) and
#             math.isnan(hist.loc[idx,'model_params'])):
                
#                 # Adjust model parameters according to particle
#                 for p in range(0,dimensions_discrete):  
#                     setattr(model,list(param_bounds.keys())[p],
#                             swarm_position[particle,p])
                
#                 model.Initialize()
                
#                 # Estimate parameters
#                 results = ModelTraining(model,data,initializations, 
#                                         BFR=False, p_opts=p_opts, 
#                                         s_opts=s_opts)
                
#                 # Save all results of this particle in a file somewhere so that
#                 # the nonlinear optimization does not have to be done again
                
#                 pkl.dump(results, open(model.name +'/' + 'particle' + 
#                                     str(swarm_position[particle]) + '.pkl',
#                                     'wb'))
                
#                 # calculate best performance over all initializations
#                 cost[particle] = results.loss.min()
                
#                 # Save new data to dictionary for future iterations
#                 hist.loc[idx,'cost'] = cost[particle]
                
#                 # Save model parameters corresponding to best performance
#                 idx_min = pd.to_numeric(results['loss'].str[0]).argmin()
#                 hist.loc[idx,'model_params'] = \
#                 [results.loc[idx_min,'params']]
                
#                 # Save DataFrame to File
#                 pkl.dump(hist, open(model.name +'/' + 'HyperParamPSO_hist.pkl'
#                                     ,'wb'))
                
#             else:
#                 cost[particle] = hist.loc[idx].cost.item()
                
        
        
        
#         cost = cost.reshape((n_particles,))
#         return cost
    
    
#     # Solve PSO Optimization Problem
#     PSO_problem.optimize(PSO_cost_function, iters=100, **cost_func_kwargs)
    
#     # Load intermediate results
#     hist = pkl.load(open(model.name +'/' + 'HyperParamPSO_hist.pkl','rb'))
    
#     # Delete file with intermediate results
#     os.remove(model.name +'/' + 'HyperParamPSO_hist.pkl')
    
#     return hist