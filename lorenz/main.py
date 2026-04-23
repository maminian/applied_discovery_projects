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

X = tools.poly_design_from_data( sol.y.T )
y = np.array([rhs(0,xi) for xi in sol.y.T]) # for now: exact derivatives.
d = y.shape[1]

lassoooo = linear_model.Lasso(alpha=1, fit_intercept=False)

lassoooo.fit(X,y)
_w = lassoooo.coef_
new_rhs = tools.ode_rhs_generator(lassoooo.coef_)
    
resol = sp.integrate.solve_ivp(new_rhs, t[[0,-1]], [1,2,3], t_eval=t)


################################

if __name__=="__main__":
    print('learned ODE:')
    from matplotlib import pyplot as plt
    
    tools.print_ode(_w, print_precision=2)

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

