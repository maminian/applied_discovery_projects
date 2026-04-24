# Lorenz...


import scipy as sp
import numpy as np
import tools

from sklearn import linear_model


def rhs(t,y, sigma=10, rho=28, beta=8/3):
    return np.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])
#

t = np.linspace(0, 20,1000)
sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], [1,2,3], t_eval = t)

h = sol.t[1] - sol.t[0]

#

####
# Design matrix
Phi = tools.poly_design_from_data( sol.y.T )


if True:
    # Direct numerical diff.
    y = np.array([rhs(0,xi) for xi in sol.y.T]) # for now: exact derivatives.
    
    y = (sol.y.T[2:] - sol.y.T[:-2])/(2*h)
    lstsq_sol = np.linalg.lstsq(Phi[1:-1], y)
    new_coef = lstsq_sol[0]
else:
    # Weak formulation.
    A = np.zeros((Phi.shape[0]-2, Phi.shape[1]))
    DU = np.zeros((Phi.shape[0]-2,3))

    for i in range(A.shape[0]):
        A[i] = h/12 * (3*Phi[i] + 10*Phi[i+1] + 3*Phi[i+2])
        DU[i] = 2/3*(sol.y.T[i+2] - sol.y.T[i])
    #
    lstsq_sol = np.linalg.lstsq(A, DU)
    new_coef = lstsq_sol[0]
#

#
# Reconstruct a new trajectory (forward error?)
new_rhs = tools.ode_rhs_generator(new_coef.T)

resol = sp.integrate.solve_ivp(new_rhs, t[[0,-1]], [1,2,3], t_eval=t)


################################

if __name__=="__main__":
    print('learned ODE:')
    # at a very coarse level, we do well:
    print(np.round(new_coef.T,1))
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

