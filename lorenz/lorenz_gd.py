import torch
import numpy as np
from torch import nn
import scipy as sp

import tools

class GD_regressor(nn.Module):
    def __init__(self, forward_solver, theta_shape):
        '''
        
        '''
        super().__init__()
        
        self.forward_solver = forward_solver
        
        # make optimizeable via nn.Parameter
        
        self.theta = nn.Parameter(torch.zeros(theta_shape), requires_grad=True)
        return

    def forward(self):
        output = 0
        return output
    #
    
    def train(self, nit=10, lr=1e-1, print_every=1):

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.lr = lr
        
        func = self()
        
        for k in range(1,nit):
            func = self()    # really: self.forward()
            func.backward()  # evaluate the gradient of self.objective at x_k
            optimizer.step() # take a step down the gradient.
            optimizer.zero_grad() # reset the internal storage for the gradient.
            
        #
        
        return self.theta
    #
#

def forward_solve(theta, ic):
    sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], ic, t_eval = t)
    return sol.y.T
#

def weakform_sol(Y):
    Phi = tools.poly_design_from_data( Y )

    # Weak formulation.
    A = np.zeros((Phi.shape[0]-2, Phi.shape[1]))
    DU = np.zeros((Phi.shape[0]-2,3))

    for i in range(A.shape[0]):
        A[i] = h/12 * (3*Phi[i] + 10*Phi[i+1] + 3*Phi[i+2])
        DU[i] = 2/3*(Y[i+2] - Y[i])
    #
    lstsq_sol = np.linalg.lstsq(A, DU)[0]
    return lstsq_sol
#

# trying to discover this.
def rhs(t,y, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])
#

t = np.linspace(0,20,1000)
sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], [1,2,3], t_eval = t)

h = sol.t[1] - sol.t[0]

#


