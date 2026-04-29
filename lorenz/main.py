# Lorenz...

import scipy as sp
import numpy as np
import tools

import pandas as pd

from sklearn import linear_model


def rhs(t,y, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])
#

t = np.linspace(0,40,2000)
sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], [1,2,3], t_eval = t)

h = sol.t[1] - sol.t[0]

#

####
# Design matrix
def paramsweep_fun(N=30):
    #N = 30 # go up until this index.
    Phi = tools.poly_design_from_data( sol.y.T )

    Phi = Phi[:N]
    Y = sol.y.T[:N]


    if False:
        # Direct numerical diff.

        y = (Y[2:] - Y[:-2])/(2*h)
        lstsq_sol = np.linalg.lstsq(Phi[1:-1], y)
        new_coef = lstsq_sol[0]
    else:
        # Weak formulation.
        A = np.zeros((Phi.shape[0]-2, Phi.shape[1]))
        DU = np.zeros((Phi.shape[0]-2,3))

        for i in range(A.shape[0]):
            A[i] = h/12 * (3*Phi[i] + 10*Phi[i+1] + 3*Phi[i+2])
            DU[i] = 2/3*(Y[i+2] - Y[i])
        #
        lstsq_sol = np.linalg.lstsq(A, DU)
        new_coef = lstsq_sol[0]
    #

    #
    # Reconstruct a new trajectory (forward error?)
    new_rhs = tools.ode_rhs_generator(new_coef.T)

    resol = sp.integrate.solve_ivp(new_rhs, t[[0,-1]], [1,2,3], t_eval=t)



    #print(np.round(new_coef.T,1))


    ################################

    #Strategy: in chunked units in time, e.g. 0-2, 2-4, etc;
    # (or maybe focus in row numbers...i=0-100; i=101-200, etc)
    # RMSE; or whatever, between, the exact (x,y,z) and the 
    # reproduced (x,y,z); understanding. 

    df = pd.DataFrame(resol.y.T - sol.y.T, columns=['x', 'y', 'z'])
    errors = [np.sqrt((row**2).sum().sum()/(3*N)) for row in df.rolling(window=N, step=N)]
    
    return errors[3] # for now
#

inputs = np.arange(30,300, dtype=int)
outputs = np.zeros(np.shape(inputs))
for i in range(len(inputs)):
    try:
        outputs[i] = paramsweep_fun(inputs[i])
        print(f'{inputs[i]}, {outputs[i]:.3}')
    except:
        outputs[i] = np.nan
        continue
    
#print(errors[:5])
#

from matplotlib import pyplot as plt
fig,ax = plt.subplots()
ax.plot(h*inputs,outputs, marker='.')
ax.grid()
ax.set(xlabel='Training time length $\Delta t$', ylabel='RMSE at $[3\Delta t,4\Delta t]$')
fig.show()

if False:
    if __name__=="__main__":
        print('learned ODE:')
        # at a very coarse level, we do well:
        
        from matplotlib import pyplot as plt
        
        #tools.print_ode(_w, print_precision=2)
        fig,ax = plt.subplots()

        ax.plot(sol.t, sol.y.T, label='original solution')
        ax.plot(resol.t, resol.y.T, label='fitted ODE solution', ls='--')
        ax.legend()
        fig.show()

        #
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot(*sol.y, label='exsact')
        ax2.plot(*resol.y, label='rebuilt')

        fig.show()
        fig2.show()

