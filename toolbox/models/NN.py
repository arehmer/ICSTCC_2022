# -*- coding: utf-8 -*-

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from ..optim.common import RK4
from .activations import *
from .layers import NN_layer, Eval_FeedForward_NN
from .initializations import (XavierInitialization, RandomInitialization, 
HeInitialization)


class recurrent():
    '''
    Parent class for all recurrent Models
    '''
    
    def ParameterInitialization(self):
        '''
        Routine for parameter initialization. Takes input_names from the Casadi-
        Function defining the model equations self.Function and defines a 
        dictionary with input_names as keys. According to the initialization
        procedure defined in self.init_proc each key contains 
        a numpy array of appropriate shape

        Returns
        -------
        None.

        '''
                
        # Initialization procedure
        if self.init_proc == 'random':
            initialization = RandomInitialization
        elif self.init_proc == 'xavier':
            initialization = XavierInitialization
        elif self.init_proc == 'he':
            initialization = HeInitialization      
        
        # Define all parameters in a dictionary and initialize them 
        self.Parameters = {}
        
        # new_param_values = {}
        for p_name in self.Function.name_in()[2::]:
            self.Parameters[p_name] = initialization(self.Function.size_in(p_name))
        
        # self.SetParameters(new_param_values)

        # Initialize with specific inital parameters if given
        # self.SetParameters(self.initial_params)
        if self.initial_params is not None:
            for param in self.initial_params.keys():
                if param in self.Parameters.keys():
                    self.Parameters[param] = self.initial_params[param]
        
        return None
                    
            
    def OneStepPrediction(self,x0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[2::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
                # params_new.append(self.Parameters[name])  
            
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
                              
        return x1,y1       
           
   
    def Simulation(self,x0,u,params=None,**kwargs):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in self.Function.name_in()[2::]:
 
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                params_new.append(self.Parameters[name])
        
        u = u[self.u_label].values
        
        F_sim = self.Function.mapaccum(u.shape[0])
        # print(params_new)
        x,y = F_sim(x0,u.T,*params_new)
        
        x = x.T
        y = y.T

        return x,y    
    
    def SetParameters(self,params):
            
        for p_name in self.Function.name_in()[2::]:
            try:
                self.Parameters[p_name] = params[p_name]
            except:
                pass      


class RBFLPV(recurrent):
    """
    Quasi-LPV model structure for system identification. Uses local linear models
    with nonlinear interpolation using RBFs. Scheduling variables are the
    input and the state.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name, dim_theta=0,
                 NN_dim=[],NN_act=[],initial_params=None, frozen_params = [],
                 init_proc='random'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        self.u_label = u_label
        self.y_label = y_label
        
        self.dim_theta = dim_theta
        self.dim = dim_theta
        self.NN_dim = NN_dim
        self.NN_act = NN_act
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
        
        self.name = 'RBF_LPV_statesched'
        
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_x 
        dim_y = self.dim_y   
        dim_theta = self.dim_theta
        NN_dim = self.NN_dim
        NN_act = self.NN_act

        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
                    
        # Define Model Parameters
        A = cs.MX.sym('A',dim_x,dim_x,dim_theta)
        B = cs.MX.sym('B',dim_x,dim_u,dim_theta)
        C = cs.MX.sym('C',dim_y,dim_x,dim_theta)
        
        # Define the scheduling map
       
        NN = []
        
        # If u and the state itself are the scheduling signals
        if len(NN_dim)==0:
            c_u = cs.MX.sym('c_u',dim_u,1,dim_theta)
            c_x = cs.MX.sym('c_x',dim_x,1,dim_theta)
            w_u = cs.MX.sym('w_u',dim_u,1,dim_theta)
            w_x = cs.MX.sym('w_x',dim_x,1,dim_theta)                
        
        # Else a NN is performing the scheduling map
        else:                
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym('NN_Wx'+str(l),NN_dim[l],dim_x),
                              cs.MX.sym('NN_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym('NN_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym('NN_b'+str(l),NN_dim[l],1)]
                    
                NN.append(params)
                
            c_h = cs.MX.sym('c_h',NN_dim[-1],1,dim_theta)
            w_h = cs.MX.sym('w_h',NN_dim[-1],1,dim_theta)
             
        # Save Neural Networks for further use
        self.NN = NN
                
        # Define Model Equations, loop over all local models
        x_new = 0
        r_sum = 0
        
        for loc in range(0,len(A)):
            
            if len(NN)==0:
                c = cs.vertcat(c_x[loc],c_u[loc])
                w = cs.vertcat(w_x[loc],w_u[loc])
                
                r = RBF(cs.vertcat(x,u),c,w)
            else:
                # Calculate the activations of the NN
                NN_out = Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act)
                
                r = RBF(NN_out[-1],c_h[loc],w_h[loc])
                
                
            x_new = x_new + \
            r * (cs.mtimes(A[loc],x) + cs.mtimes(B[loc],u)) # + O[loc])
            
            r_sum = r_sum + r
        
        x_new = x_new / (r_sum + 1e-20)
        
        y_new = 0
        r_sum = 0
        
        for loc in range(dim_theta):
            
            if len(NN)==0:
                c = cs.vertcat(c_x[loc],c_u[loc])
                w = cs.vertcat(w_x[loc],w_u[loc])
                
                r = RBF(cs.vertcat(x,u),c,w)
            else:
                # Calculate the activations of the NN
                NN_out = Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act)
                
                r = RBF(NN_out[-1],c_h[loc],w_h[loc])  
                
            y_new = y_new + r * (cs.mtimes(C[loc],x_new))
            
            r_sum = r_sum + r
            
        y_new = y_new / (r_sum + 1e-20)
        
        # Define Casadi Function
       
        # Define input of Casadi Function and save all parameters in 
        # dictionary
                    
        input = [x,u]
        input_names = ['x','u']
        
        Parameters = {}
        
        # Add local model parameters
        for loc in range(dim_theta):
            
            i=str(loc)
            
            input.extend([A[loc],B[loc],C[loc]])
            input_names.extend(['A'+i,'B'+i,'C'+i])
            
            if len(NN)==0:
                input.extend([c_u[loc],c_x[loc],w_u[loc],w_x[loc]])
                input_names.extend(['c_u'+i,'c_x'+i,'w_u'+i,'w_x'+i])
            else:
                input.extend([c_h[loc],w_h[loc]])
                input_names.extend(['c_h'+i,'w_h'+i])
                
        for l in range(0,len(NN)):
    
            input.extend(NN[l])
            k=str(l)
        
            if l==0:
                input_names.extend(['NN_Wx'+k,
                                    'NN_Wu'+k,
                                    'NN_b'+k])
          
            else:
                input_names.extend(['NN_W'+k,
                                    'NN_b'+k])   
        
        
        output = [x_new,y_new]
        output_names = ['x_new','y_new']  
        
        self.Function = cs.Function(self.name, input, output, input_names,output_names)

        # Calculate affine parameters
        # theta = XXX
        
        # self.AffineParameters = cs.Function('AffineParameters',input,
        #                                     [theta],input_names,['theta'])
        
        
        self.ParameterInitialization()
        
        return None
        
    def InitializeLocalModels(self,A,B,C,range_x=None,range_u=None):
        '''
        Initializes all local models with a given linear model and distributes
        the weighting functions uniformly over a given range
        A: array, system matrix
        B: array, input matrix
        C: array, output matrix
        op_range: array, expected operating range over which to distribute 
        the weighting functions
        '''
       
        if self.initial_params is None:
            self.initial_params = {}
       
        for loc in range(0,self.dim_theta):
            
                i = str(loc)
                self.initial_params['A'+i] = A
                self.initial_params['B'+i] = B              
                self.initial_params['C'+i] = C
                
                self.initial_params['c_u'+i] = range_u[:,[0]] + \
                    (range_u[:,[1]]-range_u[:,[0]]) * \
                        np.random.uniform(size=(self.dim_u,1))
                self.initial_params['c_x'+i] = range_x[:,[0]] + \
                    (range_x[:,[1]]-range_x[:,[0]]) * \
                        np.random.uniform(size=(self.dim_x,1))
        
        return None       
    
    def AffineStateSpaceMatrices(self,theta):
        """
        A function that returns the state space matrices at a given value 
        for theta
        """
        
        return None #A,B,C

    def AffineParameters(self,x0,u0):
        '''

        '''
        
        params = self.Parameters
        
        params_new = []
            
        for name in self.AffineParameters.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        theta = self.AffineParameters(x0,u0,*params_new) 

        return theta   
    

    
class G_RNN_new(recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN with "deep" gates that can be transformed into an affine quasi LPV
    representation. Scheduling variables are the input and the state.
    """

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,dim_thetaA=0,
                 dim_thetaB=0,dim_thetaC=0,NN_1_dim=[],NN_2_dim=[],NN_3_dim=[],
                 NN1_act=[],NN2_act=[],NN3_act=[], initial_params=None,
                 init_proc='random'):
        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        NN_1_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix
        NN_2_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the input matrix      
        NN_3_dim: list, each entry is an integer specifying the number of neurons 
        in the hidden layers of the NN associated with the system matrix     
        
        activation: list, each entry is an integer, that specifies the
        activation function used in the layers of the NNs
                    0 --> tanh()
                    1 --> logistic()
                    2 --> linear()
        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        self.NN_1_dim = NN_1_dim
        self.NN_2_dim = NN_2_dim
        self.NN_3_dim = NN_3_dim
        self.NN1_act = NN1_act
        self.NN2_act = NN2_act
        self.NN3_act = NN3_act

        self.u_label = u_label
        self.y_label = y_label
        self.name = name
        
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc
               
        self.Initialize()

    def Initialize(self):
            
        # For convenience of notation
        dim_u = self.dim_u
        dim_x = self.dim_x 
        dim_y = self.dim_y   
        dim_thetaA = self.dim_thetaA
        dim_thetaB = self.dim_thetaB
        dim_thetaC = self.dim_thetaC
        NN_1_dim = self.NN_1_dim
        NN_2_dim = self.NN_2_dim
        NN_3_dim = self.NN_3_dim    
        NN1_act = self.NN1_act
        NN2_act = self.NN2_act
        NN3_act = self.NN3_act
       
        name = self.name
        
        # Define input, state and output vector
        u = cs.MX.sym('u',dim_u,1)
        x = cs.MX.sym('x',dim_x,1)
        
        # Define Model Parameters for the linear part
        A_0 = cs.MX.sym('A_0',dim_x,dim_x)
        B_0 = cs.MX.sym('B_0',dim_x,dim_u)
        C_0 = cs.MX.sym('C_0',dim_y,dim_x)
        
        # Define Model Parameters for the time varying part by Lachhab
        A_1 = cs.MX.sym('A_1',dim_x,dim_thetaA)
        E_1 = cs.MX.sym('E_1',dim_thetaA,dim_x)
  
        B_1 = cs.MX.sym('B_1',dim_x,dim_thetaB)
        E_2 = cs.MX.sym('E_2',dim_thetaB,dim_u)

        C_1 = cs.MX.sym('C_1',dim_y,dim_thetaC)
        E_3 = cs.MX.sym('E_3',dim_thetaC,dim_x)            
        
        # Define Parameters for the multiplicative deep Neural Networks 
        NN1 = []
        NN2 = []
        NN3 = []
        
        for NN, NN_name, NN_dim in zip([NN1,NN2,NN3],['NN1','NN2','NN3'],
                                       [NN_1_dim,NN_2_dim,NN_3_dim]):
            
            for l in range(0,len(NN_dim)):
            
                if l == 0:
                    params = [cs.MX.sym(NN_name+'_Wx'+str(l),NN_dim[l],dim_x),
                              cs.MX.sym(NN_name+'_Wu'+str(l),NN_dim[l],dim_u),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                else:
                    params = [cs.MX.sym(NN_name+'_W'+str(l),NN_dim[l],NN_dim[l-1]),
                              cs.MX.sym(NN_name+'_b'+str(l),NN_dim[l],1)]
                
                NN.append(params)
        
        # Save Neural Networks for further use
        self.NN1 = NN1
        self.NN2 = NN2
        self.NN3 = NN3
        
        # Define Model Equations
       
        # Calculate the activations of the NNs by looping over each NN and
        # each layer
        NN_out = [[0],[0],[0]]
        
        for out,NN,NN_act in zip(NN_out,[NN1,NN2,NN3],[NN1_act,NN2_act,NN3_act]):
            
            out.extend(Eval_FeedForward_NN(cs.vertcat(x,u),NN,NN_act))
            
        # State and output equation
        x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_1, 
                NN_out[0][-1]*cs.tanh(cs.mtimes(E_1,x))) + cs.mtimes(B_1, 
                NN_out[1][-1]*cs.tanh(cs.mtimes(E_2,u)))
        y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_1, 
                NN_out[2][-1]*cs.tanh(cs.mtimes(E_3,x_new)))
        
        
        # Define inputs and outputs for casadi function
        input = [x,u,A_0,A_1,E_1,B_0,B_1,E_2,C_0,C_1,E_3]
        input_names = ['x','u','A_0','A_1','E_1','B_0','B_1','E_2','C_0',
                       'C_1','E_3']
        
       
        # Add remaining parameters in loop since they depend on depth of NNs
        for NN_name, NN in zip(['NN1','NN2','NN3'],[NN1,NN2,NN3]):
            
            for l in range(0,len(NN)):
            
                input.extend(NN[l])
                i=str(l)
                
                if l==0:
                    input_names.extend([NN_name+'_Wx'+i,
                                        NN_name+'_Wu'+i,
                                        NN_name+'_b'+i])
              
                else:
                    input_names.extend([NN_name+'_W'+i,
                                        NN_name+'_b'+i])
        
       
        output = [x_new,y_new]
        output_names = ['x_new','y_new']
        
        self.Function = cs.Function(name, input, output, input_names,
                                    output_names)
       
        # Calculate affine parameters
        theta_A = NN_out[0][-1] * cs.tanh(cs.mtimes(E_1,x))/cs.mtimes(E_1,x)
        theta_B = NN_out[1][-1] * cs.tanh(cs.mtimes(E_2,u))/cs.mtimes(E_2,u)
        theta_C = NN_out[2][-1] * cs.tanh(cs.mtimes(E_3,x))/cs.mtimes(E_3,x)
        
        theta = cs.vertcat(theta_A,theta_B,theta_C)   
        
        self.AffineParameters = cs.Function('AffineParameters',input,
                                            [theta],input_names,['theta'])
        
        # Initialize symbolic variables with numeric values
        self.ParameterInitialization()
        
        return None
    

    
        
    def AffineStateSpaceMatrices(self,theta):
        
        A_0 = self.Parameters['A_0']
        B_0 = self.Parameters['B_0']
        C_0 = self.Parameters['C_0']
    
        A_lpv = self.Parameters['A_0']
        B_lpv = self.Parameters['B_lpv']
        C_lpv = self.Parameters['C_lpv']  
    
        W_A = self.Parameters['W_A']
        W_B = self.Parameters['W_B']
        W_C = self.Parameters['W_C']      
    
        theta_A = theta[0:self.dim_thetaA]
        theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
                        self.dim_thetaB+self.dim_thetaC]
        
        A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return A,B,C




class G_RNN(recurrent):

    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,dim_thetaA=0,
                 dim_thetaB=0,dim_thetaC=0,fA_dim=0,fB_dim=0,fC_dim=0,
                 activation=0,initial_params=None,frozen_params =[],
                 init_proc='random'):
        '''
        Initializes the model structure by Rehmer et al. 2021.
        dim_u: int, dimension of the input vector
        dim_x: int, dimension of the state vector
        dim_y: int, dimension of the output vector
        dim_thetaA: int, dimension of the affine parameter associated with the 
        system matrix
        dim_thetaB: int, dimension of the affine parameter associated with the 
        input matrix
        dim_thetaC: int, dimension of the affine parameter associated with the 
        output matrix
        fA_dim: int, number of neurons in the hidden layer of the NN associated 
        with the system matrix
        fB_dim: int, number of neurons in the hidden layer of the NN associated 
        with the input matrix        
        fC_dim: int, number of neurons in the hidden layer of the NN associated 
        with the output matrix        
        
        activation: int, specifies activation function used in the NNs
                    0 --> tanh()
                    1 --> logistic()
                    2 --> linear()
        initial_params: dict, dictionary specifying the inital parameter values
        name: str, specifies name of the model
        '''
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        self.fA_dim = fA_dim
        self.fB_dim = fB_dim
        self.fC_dim = fC_dim
        self.activation = activation

        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
       
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc

        self.Initialize()

    def Initialize(self,initial_params=None):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_x 
            dim_y = self.dim_y   
            dim_thetaA = self.dim_thetaA
            dim_thetaB = self.dim_thetaB
            dim_thetaC = self.dim_thetaC
            fA_dim = self.fA_dim
            fB_dim = self.fB_dim
            fC_dim = self.fC_dim    
           
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            y = cs.MX.sym('y',dim_y,1)
            
            # Define Model Parameters
            A_0 = cs.MX.sym('A_0',dim_x,dim_x)
            A_lpv = cs.MX.sym('A_lpv',dim_x,dim_thetaA)
            W_A = cs.MX.sym('W_A',dim_thetaA,dim_x)
            
            W_fA_x = cs.MX.sym('W_fA_x',fA_dim,dim_x)
            W_fA_u = cs.MX.sym('W_fA_u',fA_dim,dim_u)
            b_fA_h = cs.MX.sym('b_fA_h',fA_dim,1)
            W_fA = cs.MX.sym('W_fA',dim_thetaA,fA_dim)
            b_fA = cs.MX.sym('b_fA',dim_thetaA,1)
            
            B_0 = cs.MX.sym('B_0',dim_x,dim_u)
            B_lpv = cs.MX.sym('B_lpv',dim_x,dim_thetaB)
            W_B = cs.MX.sym('W_B',dim_thetaB,dim_u)
  
            W_fB_x = cs.MX.sym('W_fB_x',fB_dim,dim_x)
            W_fB_u = cs.MX.sym('W_fB_u',fB_dim,dim_u)
            b_fB_h = cs.MX.sym('b_fB_h',fB_dim,1)
            W_fB = cs.MX.sym('W_fB',dim_thetaB,fB_dim)
            b_fB = cs.MX.sym('b_fB',dim_thetaB,1)            
  
            C_0 = cs.MX.sym('C_0',dim_y,dim_x)
            C_lpv = cs.MX.sym('C_lpv',dim_y,dim_thetaC)
            W_C = cs.MX.sym('W_C',dim_thetaC,dim_x)
            
            W_fC_x = cs.MX.sym('W_fC_x',fC_dim,dim_x)
            W_fC_u = cs.MX.sym('W_fC_u',fC_dim,dim_u)
            b_fC_h = cs.MX.sym('b_fC_h',fC_dim,1)
            W_fC = cs.MX.sym('W_fC',dim_thetaC,fC_dim)
            b_fC = cs.MX.sym('b_fC',dim_thetaC,1)            
                   
            
            # Define Model Equations
            fA_h = cs.tanh(cs.mtimes(W_fA_x,x) + cs.mtimes(W_fA_u,u) + b_fA_h)
            fA = logistic(cs.mtimes(W_fA,fA_h)+b_fA)
            
            fB_h = cs.tanh(cs.mtimes(W_fB_x,x) + cs.mtimes(W_fB_u,u) + b_fB_h)
            fB = logistic(cs.mtimes(W_fB,fB_h)+b_fB)
            
            fC_h = cs.tanh(cs.mtimes(W_fC_x,x) + cs.mtimes(W_fC_u,u) + b_fC_h)
            fC = logistic(cs.mtimes(W_fC,fC_h)+b_fC)
            
            x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_lpv, 
                    fA*cs.tanh(cs.mtimes(W_A,x))) + cs.mtimes(B_lpv, 
                    fB*cs.tanh(cs.mtimes(W_B,u)))
            y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_lpv, 
                    fC*cs.tanh(cs.mtimes(W_C,x_new)))
            
            input = [x,u,A_0,A_lpv,W_A,W_fA_x,W_fA_u,b_fA_h,W_fA,b_fA,
                      B_0,B_lpv,W_B,W_fB_x,W_fB_u,b_fB_h,W_fB,b_fB,
                      C_0,C_lpv,W_C,W_fC_x,W_fC_u,b_fC_h,W_fC,b_fC]
            
            input_names = ['x','u',
                            'A_0','A_lpv','W_A','W_fA_x','W_fA_u', 'b_fA_h',
                            'W_fA','b_fA',
                            'B_0','B_lpv','W_B','W_fB_x','W_fB_u','b_fB_h',
                            'W_fB','b_fB',
                            'C_0','C_lpv','W_C','W_fC_x','W_fC_u','b_fC_h',
                            'W_fC','b_fC']
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']
            
            self.Function = cs.Function(name, input, output, input_names,
                                        output_names)
            
            
            # Calculate affine parameters
            theta_A = fA * cs.tanh(cs.mtimes(W_A,x))/(cs.mtimes(W_A,x)+1e-6)
            theta_B = fB * cs.tanh(cs.mtimes(W_B,u))/(cs.mtimes(W_B,u)+1e-6)
            theta_C = fC * cs.tanh(cs.mtimes(W_C,x))/(cs.mtimes(W_C,x)+1e-6)
            
            theta = cs.vertcat(theta_A,theta_B,theta_C)   
            
            self.AffineParameters = cs.Function('AffineParameters',input,
                                                [theta],input_names,['theta'])
            
            self.ParameterInitialization()
                
            return None
        
    def AffineStateSpaceMatrices(self,theta):
        
        A_0 = self.Parameters['A_0']
        B_0 = self.Parameters['B_0']
        C_0 = self.Parameters['C_0']
    
        A_lpv = self.Parameters['A_lpv']
        B_lpv = self.Parameters['B_lpv']
        C_lpv = self.Parameters['C_lpv']  
    
        W_A = self.Parameters['W_A']
        W_B = self.Parameters['W_B']
        W_C = self.Parameters['W_C']      
    
        theta_A = theta[0:self.dim_thetaA]
        theta_B = theta[self.dim_thetaA:self.dim_thetaA+self.dim_thetaB]
        theta_C = theta[self.dim_thetaA+self.dim_thetaB:self.dim_thetaA+
                        self.dim_thetaB+self.dim_thetaC]
        
        A = A_0 + np.linalg.multi_dot([A_lpv,np.diag(theta_A),W_A])
        B = B_0 + np.linalg.multi_dot([B_lpv,np.diag(theta_B),W_B])
        C = C_0 + np.linalg.multi_dot([C_lpv,np.diag(theta_C),W_C]) 
        
        return A,B,C

    def EvalAffineParameters(self,x0,u0,params=None):
        '''

        '''
        if params==None:
            params = self.Parameters        
        
        params_new = []
            
        for name in self.AffineParameters.name_in():
            try:
                params_new.append(params[name])                                  # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        theta = self.AffineParameters(x0,u0,*params_new)

        return theta


class S_RNN(recurrent):
    """
    Quasi-LPV model structure for system identification. Uses a structured
    RNN which can be transformed into an affine LPV representation. Scheduling variables are the
    input and the state.
    """
    
    def __init__(self,dim_u,dim_x,dim_y,u_label,y_label,name,
                 dim_thetaA=0,dim_thetaB=0,dim_thetaC=0,
                 initial_params=None, frozen_params = [], init_proc='random'):
        
        self.dim_u = dim_u
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_thetaA = dim_thetaA
        self.dim_thetaB = dim_thetaB
        self.dim_thetaC = dim_thetaC
        
        self.dim = dim_thetaA+dim_thetaB+dim_thetaC
        
        self.u_label = u_label
        self.y_label = y_label
        self.name = name
       
        self.initial_params = initial_params
        self.frozen_params = frozen_params
        self.init_proc = init_proc

        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            dim_u = self.dim_u
            dim_x = self.dim_x 
            dim_y = self.dim_y   
            dim_thetaA = self.dim_thetaA
            dim_thetaB = self.dim_thetaB
            dim_thetaC = self.dim_thetaC
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',dim_u,1)
            x = cs.MX.sym('x',dim_x,1)
            
            # Define Model Parameters
            A_0 = cs.MX.sym('A_0_'+name,dim_x,dim_x)
            A_lpv = cs.MX.sym('A_lpv_'+name,dim_x,dim_thetaA)
            W_A = cs.MX.sym('W_A_'+name,dim_thetaA,dim_x)
            
            B_0 = cs.MX.sym('B_0_'+name,dim_x,dim_u)
            B_lpv = cs.MX.sym('B_lpv_'+name,dim_x,dim_thetaB)
            W_B = cs.MX.sym('W_B_'+name,dim_thetaB,dim_u)
            
            C_0 = cs.MX.sym('C_0_'+name,dim_y,dim_x)
            C_lpv = cs.MX.sym('C_lpv_'+name,dim_y,dim_thetaC)
            W_C = cs.MX.sym('W_C_'+name,dim_thetaC,dim_x)
           
            # Define Model Equations
            x_new = cs.mtimes(A_0,x) + cs.mtimes(B_0,u) + cs.mtimes(A_lpv, 
                    cs.tanh(cs.mtimes(W_A,x))) + cs.mtimes(B_lpv, 
                    cs.tanh(cs.mtimes(W_B,u)))
            y_new = cs.mtimes(C_0,x_new) + cs.mtimes(C_lpv, 
                    cs.tanh(cs.mtimes(W_C,x_new)))
            
            
            input = [x,u,A_0,A_lpv,W_A,B_0,B_lpv,W_B,C_0,C_lpv,W_C]  
            
            # Filter out parameters that have dimenion 0 in any direction
            input = [i for i in input if all(i.shape)]
            
            input_names = [var.name() for var in input]
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
            
            self.Function = cs.Function(name, input, output, input_names,output_names)

            # # Calculate affine parameters
            # theta_A = cs.tanh(cs.mtimes(W_A,x)/(cs.mtimes(W_A,x)))
            # theta_B = cs.tanh(cs.mtimes(W_B,u)/(cs.mtimes(W_B,u)))
            # theta_C = cs.tanh(cs.mtimes(W_C,x)/(cs.mtimes(W_C,x)))
            
            # theta = cs.vertcat(theta_A,theta_B,theta_C)   
            
            # self.AffineParameters = cs.Function('AffineParameters',input,
            #                                     [theta],input_names,['theta'])

            self.ParameterInitialization()
            
            return None