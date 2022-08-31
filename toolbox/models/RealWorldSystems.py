# -*- coding: utf-8 -*-
import casadi as cs
import numpy as np

from optim.common import RK4


class RobotManipulator():
    """
    
    """

    def __init__(self,name):
        
        self.name = name
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',2,1)
            x = cs.MX.sym('x',4,1)
            y = cs.MX.sym('y',1,1)
            
            # Sampling time
            dt = 0.05
            
            # Define Model Parameters
            a = cs.MX.sym('a',1,1)
            b = cs.MX.sym('b',1,1)
            c = cs.MX.sym('c',1,1)
            d = cs.MX.sym('d',1,1)
            e = cs.MX.sym('e',1,1)
            f = cs.MX.sym('f',1,1)
            n = cs.MX.sym('n',1,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = { 'a': np.array([[5.6794]]),
                                'b': np.array([[1.473]]),
                                'c': np.array([[1.7985]]),
                                'd': np.array([[0.4]]),
                                'e': np.array([[0.4]]),
                                'f': np.array([[2]]),
                                'n': np.array([[1]])}
            
          
            cosd = cs.cos(x[0]-x[1])
            sind = cs.sin(x[0]-x[1])

            M = cs.horzcat(a,b*cosd,b*cosd,c).reshape((2,2)).T
            g = cs.vertcat(d*np.cos(x[0]),
                           e*np.sin(x[1]))
            C = cs.vertcat(b*sind*x[3]**2 + f*x[2],
                           -b*sind*x[2]**2 + f*(x[3]-x[2]))
                           
            
            # continuous dynamics
            x_new = cs.vertcat(x[2::],
                               cs.mtimes(cs.inv(M),n*u-C-g))
            
            input = [x,u,a,b,c,d,e,f,n]
            input_names = ['x','u','a','b','c','d','e','f','n']
            
            output = [x_new]
            output_names = ['x_new']  
            
            
            f_cont = cs.Function(name,input,output,
                                 input_names,output_names)  
            
            x1 = RK4(f_cont,input,dt)
            y1 = x1[0:2]
            
            self.Function = cs.Function(name, input, [x1,y1],
                                        input_names,['x1','y1'])
            
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
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        

        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
        
        # y = y[0::10]
       
        return y
    
    
    
    
    
    
    
    
    
    
    
class RobotManipulator2():
    """
    
    """

    def __init__(self,name):
        
        self.name = name
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',2,1)
            x = cs.MX.sym('x',4,1)
            
            # Sampling time
            dt = 0.1
            
            # Define Model Parameters
            l1 = cs.MX.sym('l1',1,1)        # length link 1
            l2 = cs.MX.sym('l2',1,1)        # length link 2
            m11 = cs.MX.sym('m11',1,1)      # mass link 1
            m21 = cs.MX.sym('m21',1,1)      # mass link 2
            c1 = cs.MX.sym('c1',1,1)        # mass center link 1
            c2 = cs.MX.sym('c2',1,1)        # mass center link 2
            Iz1 = cs.MX.sym('Iz1',1,1)      # intertia link 1
            Iz2 = cs.MX.sym('Iz2',1,1)      # intertia link 2
            m0 = cs.MX.sym('m0',1,1)        # mass payload
            m12 = cs.MX.sym('m12',1,1)      # mass motor 2
            fc1 = cs.MX.sym('fc1',1,1)  
            fc2 = cs.MX.sym('fc2',1,1)
            b1 = cs.MX.sym('b1',1,1)
            b2 = cs.MX.sym('b2',1,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = { 'l1': np.array([[0.45]]),
                                'l2': np.array([[0.45]]),
                                'm11': np.array([[23.902]]),
                                'm21': np.array([[1.285]]),
                                'c1': np.array([[0.091]]),
                                'c2': np.array([[0.048]]),
                                'Iz1': np.array([[1.266]]),
                                'Iz2': np.array([[0.093]]),
                                'm0': np.array([[0]]),
                                'm12': np.array([[0]]),
                                'fc1': np.array([[7.17]]),
                                'fc2': np.array([[1.734]]),
                                'b1': np.array([[2.288]]),
                                'b2': np.array([[0.175]]),
                                }
            
            
            a = l1*(m21*c2 + m0*l2)
            b = Iz2 + m0*l2**2 + m21*c2**2
            
            # Mass Inertia MAtrix
            D11 = 2*a*np.cos(x[1]) + Iz1 + Iz2 + m11*c1**2 + m12*l1**2 \
                + m21*(c2**2 + l1**2) + m0*(l1**2+l2**2)
            D12 = a*np.cos(x[1]) + b
            D21 = a*np.cos(x[1]) + b
            D22 = b
            
            C11 = -a*x[3]*np.sin(x[1])
            C12 = -a*(x[2]+x[3])*np.sin(x[1])
            C21 = a*x[2]*np.sin(x[1])
            C22 = 0
            
            G1 = ((m21+m12+m0)*l1 + m11*c1)*np.cos(x[0]) \
                +(m21*c2+m0*l2)*np.cos(x[0]+x[1])
            G2 = (m21*c2+m0*l2)*np.cos(x[0]+x[1])
          
            # Friction Term
            F1 = b1 * x[2] + fc1*cs.sign(x[2])
            F2 = b2 * x[3] + fc2*cs.sign(x[3])
          
            M = cs.horzcat(D11,D12,D21,D22).reshape((2,2)).T
            C = cs.horzcat(C11,C12,C21,C22).reshape((2,2)).T
            G = cs.vertcat(G1,
                           G2)
            F = cs.vertcat(F1,
                           F2)                           
            
            # continuous dynamics
            x_new = cs.vertcat(x[2::],
                               cs.mtimes(cs.inv(M),u-cs.mtimes(C,x[2::])-G-F))
            
            input = [x,u,l1,l2,m11,m21,c1,c2,Iz1,Iz2,m0,m12,fc1,fc2,b1,b2]
            input_names = ['x','u','l1','l2','m11','m21','c1','c2','Iz1',
                           'Iz2','m0','m12','fc1','fc2','b1','b2']
            
            output = [x_new]
            output_names = ['x_new']  
            
            
            f_cont = cs.Function(name,input,output,
                                 input_names,output_names)  
            
            x1 = RK4(f_cont,input,dt)
            y1 = x1[0:2]
            
            self.Function = cs.Function(name, input, [x1,y1],
                                        input_names,['x1','y1'])
            
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
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        

        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
        
        # y = y[0::10]
       
        return y    

class Bioreactor():
    """
    Implementation of Bioreactor described in dissertation by Verdult 
    on page 218
    """

    def __init__(self,name):
        
        self.name = name
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',1,1)
            x = cs.MX.sym('x',2,1)
            
            # Sampling Time
            T = 0.01
            
            # Define Model Parameters
            G = cs.MX.sym('G',1,1)
            b = cs.MX.sym('b',1,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = { 'G': np.array([[0.48]]),
                                'b': np.array([[0.02]])}
          
            # discrete dynamics
            
            x1_new = x[0] + T * ( -x[0]*u  + x[0]*(1-x[1]) * cs.exp(x[1]/G) ) 
            x2_new = x[1] + T * ( -x[1]*u + x[0]*(1-x[1])*cs.exp(x[1]/G) * ((1+b)/(1+b-x[1])) )
            
            x_new = cs.vertcat(x1_new,x2_new)
            y_new = x_new[0]
            
            input = [x,u,G,b]
            input_names = ['x','u','G','b']
            
            output = [x_new,y_new]
            output_names = ['x_new','y_new']  
          
            self.Function = cs.Function(name, input, output,
                                        input_names, output_names)
            
            # continuous dynamics
            
            
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
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        

        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
    
        return y