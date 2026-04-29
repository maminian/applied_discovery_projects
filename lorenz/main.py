# Lorenz...

import scipy as sp
import numpy as np
import tools

import pandas as pd

from sklearn import linear_model


def rhs(t,y, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])
#

t = np.linspace(0,20,1000)
sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], [1,2,3], t_eval = t)

h = sol.t[1] - sol.t[0]

#

####
# Design matrix
def paramsweep_fun(N=30):
    #N = 30 # go up until this index.
    
    # Question: for an *ensemble* of simulated initial conditions, 
    # how does the solution of the aggregated problem do in simulating 
    # 3 training periods out for an unseen initial condition?
    # Right now: we're "stacking" weak-form systems followed by an OLS solve 
    # to do this. Might not be the best way!
    As = []
    DUs = []
    for _ in range(10):
        sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], np.random.normal([1,1,1],4), t_eval = t)

        Phi = tools.poly_design_from_data( sol.y.T )

        Phi = Phi[:N]
        Y = sol.y.T[:N]

        # Weak formulation.
        A = np.zeros((Phi.shape[0]-2, Phi.shape[1]))
        DU = np.zeros((Phi.shape[0]-2,3))

        for i in range(A.shape[0]):
            A[i] = h/12 * (3*Phi[i] + 10*Phi[i+1] + 3*Phi[i+2])
            DU[i] = 2/3*(Y[i+2] - Y[i])
        #
        As.append(A)
        DUs.append(DU)
    #
    
    A_stack = np.vstack(As)
    DU_stack = np.vstack(DUs)
    
    lstsq_sol = np.linalg.lstsq(A_stack, DU_stack)
    new_coef = lstsq_sol[0]
    
    #
    # Reconstruct a new trajectory (forward error?)
    new_rhs = tools.ode_rhs_generator(new_coef.T)

    # Samuel gave me this initial condition.
    sol2 = sp.integrate.solve_ivp(rhs, t[[0,-1]], [4,13,17], t_eval = t)
    resol = sp.integrate.solve_ivp(new_rhs, t[[0,-1]], [4,13,17], t_eval=t)

    #print(np.round(new_coef.T,1))


    ################################

    #Strategy: in chunked units in time, e.g. 0-2, 2-4, etc;
    # (or maybe focus in row numbers...i=0 to N-1; i=N to 2N-1, etc)
    # RMSE; or whatever, between, the exact (x,y,z) and the 
    # reproduced (x,y,z); understanding. 

    # TODO: cleaner implementation
    df = pd.DataFrame(resol.y.T - sol2.y.T, columns=['x', 'y', 'z'])
    errors = [np.sqrt((row**2).sum().sum()/(3*N)) for row in df.rolling(window=N, step=N)]
    
    # success metric for now; note errors[1] 
    # theoretically covers the training period plus one
    return errors[3] 
#

inputs = np.arange(30,150, dtype=int) # expand to 300 if you want; takes some time
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
ax.set(xlabel=r'Training time length $\Delta t$', ylabel=r'RMSE at $[3\Delta t,4\Delta t]$')
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

