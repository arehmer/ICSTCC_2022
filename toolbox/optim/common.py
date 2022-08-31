# -*- coding: utf-8 -*-

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np


def OptimValues_to_dict(optim_variables_dict,sol):
    # reads optimized parameters from optim solution and writes into dictionary
    
    values = {}
    
    for key in optim_variables_dict.keys():
       dim0 = optim_variables_dict[key].shape[0]
       dim1 = optim_variables_dict[key].shape[1]
       
       values[key] = sol.value(optim_variables_dict[key]) 
       
       # Convert tu numpy array
       values[key] = np.array(values[key]).reshape((dim0,dim1))

      
    return values

def RK4(f_cont,input,dt):
    '''
    Runge Kutta 4 numerical intergration method

    Parameters
    ----------
    f_cont : casadi function
        DESCRIPTION.
    dt : int
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    k1 = f_cont(*input)
    k2 = f_cont(*[input[0]+dt*k1/2,*input[1::]]) 
    k3 = f_cont(*[input[0]+dt*k2/2,*input[1::]])
    k4 = f_cont(*[input[0]+dt*k3,*input[1::]])
    
    x_new = input[0] + 1/6 * dt * (k1+2*k2+2*k3+k4)
    
    return x_new

def MSE(y_target,y_est):
    '''
    Calculates Mean Squared Error

    Parameters
    ----------
    y_target : numpy array of shape (N,dim_y)
        Measured / true system output.
    y_est : numpy array of shape (N,dim_y)
        Estimated ouput by model

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    MSE : float
        Mean Squared Error

    '''
    
    
    if y_target.shape[0]==y_est.shape[0] and y_target.shape[1]==y_est.shape[1]:
        N = y_target.shape[0]
        dim_y = y_target.shape[1]
        
    else:
        raise ValueError('y_target and y_est need to have the same shape.')
    
    MSE = float(1/N*sum(sum((y_target-y_est)**2)))
    
    return MSE

def AIC(y_target,y_est,n,p=2):
    '''
    Calculates AIC for SISO System

    Parameters
    ----------
    y_target : numpy array of shape (N,dim_y)
        Measured / true system output.
    y_est : numpy array of shape (N,dim_y)
        Estimated ouput by model
    n : int
        Number of model parameters.
    p : int, optional
        AIC Hyperparameter. The default is 2.

    Returns
    -------
    AIC : float
        Value of AIC.

    '''

    
    mse = MSE(y_target,y_est) 
    
    N = y_target.shape[0]
    
    AIC = float(N*np.log(mse) + p*n)
    
    return AIC


def BestFitRate(y_target,y_est):
    BFR = 1-sum((y_target-y_est)**2) / sum((y_target-np.mean(y_target))**2) 
    
    BFR = BFR*100
    
    if BFR<0:
        BFR = np.array([0])
        
    return BFR