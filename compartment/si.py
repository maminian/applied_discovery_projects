import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

####

import pysindy
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate,linalg,optimize

import sys
sys.path.append('../')
import tools

# takes 2D data array X and 1D time array t; outputs Xdot of same shape as X.
differentiator = pysindy.differentiation.SINDyDerivative(kind="spline", s=1e-2)


# synthetic data
beta=0.2
gamma=0.1
x0=[0.98,0.02,0]

W_exact = np.zeros((3,10))
W_exact[0,5] = -beta
W_exact[1,5] = beta
W_exact[1,2] = -gamma
W_exact[2,2] = gamma

rhs = tools.ode_rhs_generator(W_exact)

t=np.linspace(0,100,201)
sol=integrate.solve_ivp(rhs, t[[0,-1]], x0, t_eval=t)
X = sol.y.T

n,d = np.shape(X)
#####

sigma=tools._poly_exp_basis_func_count(d, maxdegree=2)

# constrain rows of W sum to zero.
C = np.hstack([np.identity(sigma) for _ in range(d)])
Crhs = np.zeros((sigma,1))
#for i in range(d):
#    Aineq1[i,(sigma*i):(sigma*(i+1))] = 1
#

# constrain the equations to be satisfied exactly.
Phi = tools.poly_design_from_data(X)
Phi_tilde = linalg.block_diag(*[Phi for _ in range(d)]) # why is block_diag designed this way..?
Phi_rhs = differentiator(X,t)
Phi_tilde_rhs = np.reshape(Phi_rhs.T, (n*d,1))

# Join equality constraints.
#Aeq = np.vstack([C,Phi_tilde])
#beq = np.vstack([Crhs,Phi_tilde_rhs])
Aeq = Phi_tilde
beq = Phi_tilde_rhs
Aeq = np.hstack([Aeq, np.zeros((Aeq.shape[0], sigma*d))])

# Inequality constraints associated with slack variables for l1 objective.
Aineq = np.zeros((2*sigma*d, 2*sigma*d))
bineq = np.zeros((2*sigma*d,1))
for i in range(sigma*d):
    Aineq[2*i,i] = 1
    Aineq[2*i,i+sigma*d] = -1
    Aineq[2*i+1,i] = -1
    Aineq[2*i+1,i+sigma*d] = -1


# objective
objective = np.concatenate([np.zeros(sigma*d), np.ones(sigma*d)])

# solve linear program.
sol = optimize.linprog(objective, A_ub=Aineq, b_ub=bineq, A_eq=Aeq, b_eq=beq, bounds=(None,None))

