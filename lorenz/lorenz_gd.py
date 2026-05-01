import torch
import numpy as np
from torch import nn
import scipy as sp

import tools

class GD_regressor(nn.Module):
    def __init__(self, theta_shape):
        '''
        
        '''
        super().__init__()
        
        #self.forward_solver = forward_solver
        
        # make optimizeable via nn.Parameter
        
        self.theta = nn.Parameter(torch.zeros(theta_shape), requires_grad=True)
        
        self.shape = theta_shape # number of basis funs by number of dims.
        return

    def forward(self):
        # a. Random initial condition
        #initial_cond = 10*torch.rand(self.shape[0])
        #import pdb
        #pdb.set_trace()
        
        initial_cond = np.random.uniform(0,10, self.shape[0])
        y_exact = sp.integrate.solve_ivp(rhs, t[[0,-1]], initial_cond, t_eval = t)
        y_exact = y_exact.y.T
        
        y_exact = torch.tensor(y_exact)
        #
        # find theta and re-simulate.
        #theta = weakform_sol(y_exact)
        _t = self.theta.detach().numpy()
        y_approx = forward_solve(_t, initial_cond)
        
        y_approx = torch.tensor(y_approx)#.detach()
        
        # evaluate a loss....
        # BIG asterisks about this.
        #import pdb
        #pdb.set_trace()
        
        output = torch.norm(y_exact - y_approx, 2)
        output = output + torch.norm(self.theta, 1)
        
        
        #self.theta_curr = theta # save for later.
        
        return output
    #
    
    
    def weakform_sol(self,Y):
        Phi = tools.poly_design_from_data( Y )

        # Weak formulation.
        A = np.zeros((Phi.shape[0]-2, Phi.shape[1]))
        DU = np.zeros((Phi.shape[0]-2,3))

        for i in range(A.shape[0]):
            A[i] = h/12 * (3*Phi[i] + 10*Phi[i+1] + 3*Phi[i+2])
            DU[i] = 2/3*(Y[i+2] - Y[i])
        #
        lstsq_sol = torch.linalg.lstsq(torch.tensor(A), torch.tensor(DU))[0]
        return lstsq_sol
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
            
            if k%print_every==0:
                print(k, func)
        #
        
        
        
        return self.theta
    #
#

def forward_solve(theta, ic):
    new_rhs = tools.ode_rhs_generator(theta)
    sol = sp.integrate.solve_ivp(new_rhs, t[[0,-1]], ic, t_eval = t)
    return sol.y.T
#

# trying to discover this.
def rhs(t,y, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])
#

#########

t = np.linspace(0,20,1000)
sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], [1,2,3], t_eval = t)

h = sol.t[1] - sol.t[0]

#
blah = GD_regressor([3,10])
blah.train(nit=1000, lr=1e-4, print_every=100)


