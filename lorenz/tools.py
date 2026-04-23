import numpy as np
import scipy as sp
import itertools

def _poly_exp_basis_func_count(n, maxdegree=2):
    '''
    Formula for number of unique expansion terms up to degree maxdegree.
    
    Input: n: integer, number of dimensions
           maxdegree: integer, highest degree (default: 2)
    Output: p: integer, sum of appropriate combinations.
    
    Input/outputs pairs:
        (1,2) -> 3
        (2,2) -> 6
        (3,2) -> 10
        (3,1) -> 4
        (*,0) -> 1
    '''
    return sum([sp.special.comb(n,i,exact=True,repetition=True) for i in range(maxdegree+1)])

def _power_array(n, maxdegree=2):
    '''
    array of powers for original variables. Helper function to create the 
    ODE operator function.
    
    Inputs:
        n: integer, number of variables
        maxdegree: integer, maximum power (default: 2)
    Outputs:
        pow: numpy array of integers shape (n, p)
    '''
    sigma = _poly_exp_basis_func_count(n, maxdegree)
    pow = np.zeros((sigma,n), dtype=int)
    #print(pow.shape)
    #x^3, x^2y, x^2z, xyz, xy^2, y^3, y^2z, xz^2, yz^2, z^3
    idx=0
    # ascend from lower to higher-degree polynomial terms
    for deg in range(maxdegree+1):
        # identify nonzero power combinations of the given degree
        for multipow in itertools.combinations_with_replacement(range(n), deg):
            # append.
            for k in multipow:
                pow[idx,k] += 1
            idx += 1
    return pow

def print_ode(M, varnames=None, nz_thresh=1e-3, maxdegree=2, print_precision=3):
    '''
    Given coefficient matrix R, print the ODE being solved in plaintext.
    
    Inputs:
        R : numpy array; shape (n,sigma); coefficients
        varnames : list-like of strings to use for variable names;
            len(varnames)==p. If None (default), letters "x_0" through "x_{p-1}" are used.
        nz_thresh : float; terms with abs(R)<nz_tresh are not printed. (default: 1e-3)
        maxdegree : max poly degree. should be consistent with shape of M. (default: 2)
        
        print_precision : digits past decimal point for printing only (default: 3)
            NOT IMPLEMENTED. WILL BE 3 DIGITS TOTAL REGARDLESS
    '''
    n,sigma = np.shape(M)
    vn = varnames
    if vn is None:
        vn = [f'x_{i}' for i in range(sigma)]
    
    powers = _power_array(n, maxdegree=maxdegree)
    for i in range(n):
        lhs = f'd{vn[i]}/dt = '
        rhs = []
        for j in range(sigma):
            if abs(M[i,j]) < nz_thresh: 
                continue
            #suffix = f'{:.print_precision}' # TODO: how to implement this?
            term = f'{M[i,j]:.2}'
            for k in range(n):
                if powers[j,k]==0:
                    continue
                term += f'{vn[k]}^{powers[j,k]}'
            rhs.append(term)
        # end for
        
        rhs = ' + '.join(rhs) # plusses between terms
        print(lhs,rhs)
    return


def get_basis_funcs(n, maxdegree=2, varnames=None, sep=', '):
    '''
    Return list of basis functions; in particular, the order they appear in coefficient matrices.
    
    Inputs:
        d : integer number of variables/dimensions
        maxdegree : integer, highest poly degree
        varnames : list-like of strings to use for variable names;
            len(varnames)==p. If None (default), letters "x_0" through "x_{p-1}" are used.
        sep : string used to separate terms (default: ", ")
    Outputs:
        basis : list of strings of basis functions.
    
    TODO: have a "latex" switch set up to render subscripts properly.
    '''
    vn = varnames
    sigma = _poly_exp_basis_func_count(n, maxdegree)
    if vn is None:
        vn = [f'x_{i}' for i in range(sigma)]
    
    powers = _power_array(n, maxdegree=maxdegree)
    basis = []
    for j in range(sigma):
        term = ''
        for k in range(n):
            if powers[j,k]==0:
                continue
            term += f'{vn[k]}^{powers[j,k]}'
        if len(term)==0:
            term='1'
        basis.append(term)
    # end for
    
    return basis

def ode_rhs_generator(M, maxdegree=2):
    '''
    inputs:
        M: numpy array; coefficients of right-hand-side of system.
            expected shape (n,d). See below for explanation.
        maxdegree: integer, max degree of polynomials. Must conform with 
            shape of M. See below. 
    outputs:
        rhs: FUNCTION with inputs conforming the needs for scipy.integrate.solve_ivp;
            takes array shape (n,) in and out.
    
    For maxdegree=2, an ansatz is used for the right-hand-side of 
    all terms in a system of "n" ODEs. With two variables,
    the basis functions on each term would be:
    
    [x^0 y^0, x^1 y^0, x^0 y^1, x^2 y^0, x^1 y^1, x^0 y^2],
    
    a total of seven terms. Hence, R would be shape (2,6) here.
    
    With three variables and maxdegree=2, this expands to (3,10). 
    '''
    # There's a disconnect somewhere if the dimensions don't match expectations here.
    n,sigma = np.shape(M)
    #assert sigma == _poly_exp_basis_func_count(n, maxdegree=d) # really needed?
    
    def rhs_func(_t, _X):
        powers = _power_array(n, maxdegree)
        byterm = (_X**powers).T # shape (d,p); termwise x_{ij}**powers_{ij}
        
        # collapse multinomial product into actual terms, and multiply by per-row coeffs
        byterm = M * np.prod(byterm, axis=0) # shape (d,p); note multinomial values are same in every row.
        
        # sum all multinomial terms per-variable
        result = np.sum(byterm, axis=1) # shape (n,)
        return result
    
    return rhs_func

def poly_design_from_data(data, maxdegree=2):
    '''
    construct design matrix from an array of observations.
    Each row of design matrix constructs the polymonomial basis functions 
    up to degree maxdegree; mapping R^n to R^sigma, where sigma is 
    calculated from _poly_exp_basis_func_count().
    
    Ordering of basis functions is consistent with other functions in tools.py
    and can be pulled numerically from _power_array() or in plaintext 
    from get_basis_funcs().
    
    Similar to pysindy.PolynomialLibrary() and sklearn.preprocessing.PolynomialFeatures().
    
    Input: data; array of shape (N,n); N observations in n dimensions
    Output: D; numpy.array of shape (n,sigma).
    
    Example:
        X = np.array([[1,2], [3,4], [0,1]])
        D = poly_design_from_data(X, 2)
        
    D: [[1, 1, 2, 1,  2,  4],
        [1, 3, 4, 9, 12, 16],
        [1, 0, 1, 0,  0,  1]]
    '''
    N,n = np.shape(data) # N datapoints in n dimensions
    p = _poly_exp_basis_func_count(n, maxdegree=maxdegree)
    #D = torch.zeros((n,p))
    pa = _power_array(n, maxdegree)
    
    D = np.array([np.prod(data**(pa[i]),axis=1) for i in range(pa.shape[0])]).T
    return D
#

def stlsq(A,b, thresh=1e-3, maxit=10):
    # TODO: need to make work with matrix b.
    m,n = np.shape(A)
    _,p = np.shape(b)
    
    #active = np.arange(n, dtype=int)
    
    x = np.zeros((n,p), dtype=float)
    active = np.zeros((n,p), dtype=bool)
    
    for i in range(maxit):
        import pdb
        pdb.set_trace()
        # solve system limited to active set of columns of A.
        sol = np.linalg.lstsq(A[:,active],b, rcond=None)
        xtemp = sol[0]
        mask = abs(xtemp)<thresh
        if mask.sum()==0:
            break
        xtemp[mask]=0
        x[active]=xtemp
        
        active = np.setdiff1d(active, np.where(mask)[0])
        print(active)
        
    return x

def lse(A, b, B, d):
    """
    Equality-contrained least squares.
    The following algorithm minimizes ||Ax - b|| subject to the
    constraint Bx = d.
    
    Parameters
    ----------
    A : array-like, shape=[m, n]
    B : array-like, shape=[p, n]
    b : array-like, shape=[m]
    d : array-like, shape=[p]
    
    Reference
    ---------
    Matrix Computations, Golub & van Loan, algorithm 12.1.2
    
    Examples
    --------
    >>> A = np.array([[0, 1], [2, 3], [3, 4.5]])
    >>> b = np.array([1, 1])
    >>> # equality constrain: ||x|| = 1.
    >>> B = np.ones((1, 3))
    >>> d = np.ones(1)
    >>> lse(A.T, b, B, d)
    array([-0.5,  3.5, -2. ])
    
    Implementation due to Fabian Pedregosa --
        https://gist.github.com/fabianp
        https://gist.github.com/fabianp/915461
    """
    from scipy import linalg
    if not hasattr(linalg, 'solve_triangular'):
        # compatibility for old scipy
        solve_triangular = linalg.solve
    else:
        solve_triangular = linalg.solve_triangular
    A, b, B, d = map(np.asanyarray, (A, b, B, d))
    p = B.shape[0]
    Q, R = linalg.qr(B.T)
    y = solve_triangular(R[:p, :p].T, d)
    A = np.dot(A, Q)
    z = linalg.lstsq(A[:, p:], b - np.dot(A[:, :p], y))[0].ravel()
    return np.dot(Q[:, :p], y) + np.dot(Q[:, p:], z)
    

def diag_concat(arrs, sparse=False):
    '''
    Implements an array version of np.diag(...)
    where the matrix is defined by a collection 
    of matrices on the diagonal.
    
    Inputs:
        arrs: list of square 2D numpy arrays.
        sparse: boolean; if True, forms A as a sparse matrix 
            (TODO: NOT IMPLEMENTED)
    Outputs:
        A: 2D array of appropriate size with given contents.
    '''
    nrows = [np.shape(_a)[0] for _a in arrs]
    ncols = [np.shape(_a)[1] for _a in arrs]
    
    M,N = sum(nrows),sum(ncols)
    A = np.zeros((M,N))
    i,j=0,0
    for _m,_n,_a in zip(nrows,ncols,arrs):
        A[i:i+_m, j:j+_n] = _a
        i+=_m
        j+=_n
    return A

def basis_pursuit_linear_conservation(A,b,Ceq,beq):
    '''
    minimize   ||x||_1
    subject to A@x = b
               Ceq@x - beq = 0
    '''
    from scipy import optimize, linalg
    if True:
        w_proposed = lse(A, b, Ceq, beq)
        x_proposed = w_proposed
    else:
        moo = np.linalg.lstsq(A,b, rcond=None)
        x_proposed = moo[0]
    
    
    
    # TODO: can be made more efficient understanding the A and C 
    # we're forming are block diagonal; or otherwise structured.
    a_null = linalg.null_space(A)
    ceq_null = linalg.null_space(Ceq)
    
    # Set up a linear program to find a sparser 
    # solution than the OLS one, via L_1 minimization
    B = a_null @ (a_null.T @ ceq_null)
    dp1,dp2 = np.shape(B)
    
    objective = np.concatenate([np.zeros(dp2), np.ones(dp1)])
    Aineq = np.block([[B, -np.identity(dp1)], [-B, -np.identity(dp1)]])
    bineq = np.concatenate([-x_proposed, x_proposed])
    
    #Aeq = np.block([Ceq@a_null, np.zeros((dp1,dp1))])
    #beq = -Ceq@moo
    
    # TODO: explore cvxopt as an option since some of the difficulty 
    # has been passed off to the LP solver.
    lp_result = optimize.linprog(objective, A_ub=Aineq, b_ub=bineq, bounds=(None,None))
    #lp_result = optimize.linprog(objective, A_ub=Aineq, b_ub=bineq, bounds=(None,None), 
    #    method='highs-ds'
    #)
    
    #print(objective.shape, Aineq.shape, bineq.shape)
    
    # forcing integrality of slack constraints looks like it does switch this to 
    # effectively an integer program; impractical.
    #lp_result = optimize.linprog(objective, A_ub=Aineq, b_ub=bineq, bounds=(None,None),
    #    integrality=np.concatenate([np.zeros(dp2, dtype=int), np.ones(dp1, dtype=int)])
    #)
    
    z = lp_result.x[:dp2]
    x_sparse = x_proposed + B@z
    #return x_sparse
    return lp_result, x_sparse

def sparse_conserved_solve(Phi,Y, thresh=1e-2, print_errors=False):
    '''
    FRAGILE -- NOT A GENERAL PURPOSE TOOL; 
    ONLY MEANT FOR SPEPI-STYLE PROBLEMS.
    MAY GENERALIZE/POLISH IN THE FUTURE.
    
    Gives a solution to min_W ||Phi@W - Y||_1
        (strictly: 1-norm of the flattened residual)
        subject to a linear conservation constraint
        on array W (sum of each row is 0)
    
    Inputs:
        Phi : design matrix; shape (N, sigma)
        Y : derivative matrix; shape (N, n)
        thresh : relative threshold to zero out entries of W.
    
    Outputs:
        W: shape (n,sigma)
        
    Flattens W and Y and creates block diagonal version of Phi. 

    With these, the approach is:
    
    a) Solve the linearly-constrained least squares 
        for a viable solution; 
    
        min ||Phi@w - y||_2 
        s.t. A@w = 0
        
    b) Build a matrix B = null(Phi) \\cap null(A)
        (subspace of viable solutions)
    
    c) Solve the unconstrained L_1 minimization
    
        min_z ||w + Bz||_1
    
        which is handled by writing as a linear 
        program via slack variables, then calling 
        scipy.optimize.linprog.
    '''
    N,sigma = np.shape(Phi)
    _,d = np.shape(Y)
    
    Phi_block = diag_concat([Phi for _ in range(d)])
    yflat = np.concatenate(Y.T)
    
    # Column sum conservation constraints
    Ceq = np.tile(np.identity(sigma), (1,d))
    beq = np.zeros(sigma)

    #import pdb
    #pdb.set_trace()
    
    lp_result, w_result = basis_pursuit_linear_conservation(Phi_block, yflat, Ceq, beq)

    ##
    # Diagnostics
    #
    if print_errors:
        print(np.linalg.norm(yflat - Phi_block @ w_proposed))
        print(np.quantile(lp_result.x[dp2:],[0,0.1,0.5,0.9,1]))
        print(np.linalg.norm(yflat - Phi_block @ w_sparse))
        print(np.linalg.norm(w_proposed - w_sparse)/np.linalg.norm(w_proposed))
    
    w_result = np.reshape(w_result,(d,sigma))
    return w_result
#
    
    

def _basispursuit_real(A,b, xi=None):
    '''
    For rank-deficient A (nontrivial nullspace), finds a solution to
    
    min ||x||_1
    subject to Ax=b.
    
    If an initial candidate xi is provided, then only the nullspace of A 
    is sampled and the constraint is relaxed to Ax = A(xi), and there 
    is no consideration of the right-hand side b:
    
    min ||x||_1
    subject to Ax=A(xi).
    
    Inputs:
        A: array, shape (m,n); assumed m<=n; real-valued (TODO: is m<=n needed?)
        b: array, shape (m,1); real-valued
        xi: array, shape (n,1); real-valued. (default:  None). If None, then 
            an ordinary least squares solution of Ax=b is computed.
    Outputs:
        x: solution to above objective, shape (n,1); real-valued
    '''
    import scipy
    m,n = np.shape(A)
    
    if xi is None:
        # get a 2-norm least squares sol
        candidate_l2 = np.linalg.lstsq(A, b, rcond=None)
        xi = candidate_l2[0]
    else:
        assert np.shape(xi)==(n,1)
    
    ##
    # form a linear program to seek an adjustment from the nullspace of 
    # the matrix to produce a sparser solution (pure 1-norm optimization)
    curlyN = scipy.linalg.null_space(A)
    dimnullA = curlyN.shape[1]
    objective_dim = dimnullA + n

    # objective
    c = np.concatenate([np.zeros(dimnullA), np.ones(n)])

    # inequality constraints for slack vars.
    Aineq = np.block( 
    [
        [curlyN,  -np.identity(n)], 
        [-curlyN, -np.identity(n)]
    ])
    bineq = np.vstack([-xi, xi])

    ##
    # Solve and extract solution
    sol = scipy.optimize.linprog(c, A_ub=Aineq, b_ub=bineq)
    eta = curlyN @ sol.x[:dimnullA]
    eta = np.reshape(eta, (n,1))

    x = xi + eta
    return x
#

def weak_form_solve_scalar(t, x, sigma=4, r=8):
    '''
    Implementation of a weak formulation for finding weights in 
    
        \\dot{x} = \\sum_j w_j \\phi_j(x)
    
    where \\phi_j(x) = x**j, j=0, 1, ..., sigma-1.
    
    test functions \\psi are C^0 tent functions with 
    a convention of right-continous derivative hat function.
    With these choices, problem turns into a particular running sum.
    
    Input:
        t : array of time points; current expects uniformly spaced
        x : array of observed data; len(x) == len(t).
        sigma : integer; number of monomial basis functions 
            (default: 4; i.e. up to x**3)
        r : integer; radius in units of dt for the test functions. 
            r=1 corresponds to non-overlapping test functions.
            Note you need len(x) > 2*r.
            (default: r=8).
    
    Output:
        W : array (1d) of length sigma of fit coefficients.
    '''
    
    if len(x.flatten())<= 2*r:
        print('WARNING: need len(x)>2r so that there are enough test functions in the interior of the time domain.')
    d=1
    dt = t[1] - t[0]
    
    B = np.zeros((d,len(t)))
    A = np.zeros((sigma, len(t)))

    # raw design matrix
    V = np.array([x.flatten()**j for j in range(sigma)])
    # integrated (for use with triangle test functions; see scratch work)
    V_int = np.cumsum(dt*V, axis=1)

    # build A; integrals of phi_j times psi_k.
    for j in range(sigma):
        for k in range(r,len(t)-r):
            #lo = max(0,k-r)
            #hi = min(V_int.shape[1], k+r)
            A[j,k] = np.sum(V_int[j,k:(k+r)]) - np.sum(V_int[j,(k-r):k]) # todo: vectorize/streamline
    #A *= dt

    # build B; integrals of x_j times psi_k.
    for i in range(d):
        for k in range(r,len(t)-r):
            #lo = max(0,k-r)
            #hi = min(V_int.shape[1], k+r)
            B[i,k] = (np.sum(x[i,k:(k+r)]) - np.sum(x[i,(k-r):k]))
    #B *= dt

    #return A,B

    #sol = np.linalg.lstsq(A.T,B.T, rcond=None)
    #W_lstsq = sol[0].T
    #print('lstsq:',W_lstsq)
    #from sklearn import linear_model
    #arr = linear_model.Ridge()
    #arr.fit(A.T, B.T)
    #W = arr.coef_
    #_A = np.reshape(A, )
    W = _basispursuit_real(A.T, B.T)
    W = W.T
    return W
#

def pysindy_sr3_constrained_solve(t, X, maxdeg=2):
    '''
    Implements constrained optimization via pysindy's framework.
    
    Uses pysindy.ConstrainedSR3() with a constraint requiring the sum 
    of coefficients. Current version hard-codes parameter choices for 
    the constrained_sr3
    See: https://pysindy.readthedocs.io/en/latest/_modules/pysindy/optimizers/constrained_sr3.html
    
    Inputs:
        X : array, shape (N,d), observation timeseries of d state variables.
        t : array, shape (N,), of associated times.
        maxdeg : integer, maximum degree for polynomial basis expansion.
    '''
    import pysindy
    #maxdeg=2 # polynomial library max degree
    #d=3      # number of state variables
    N,d = np.shape(X)

    # Get an initial guess from an OLS solution.
    Phi = poly_design_from_data(X, maxdegree=maxdeg)
    sigma = np.shape(Phi)[1]

    # set up initial guess for W for constrained_SR3.
    # TODO: revisit unconstrained guess.
    W_0 = np.zeros((d, sigma))

    # constraints: sums of terms across compartments add to zero.
    constr_lhs = np.tile(np.identity(sigma), d)
    constr_rhs = np.zeros(sigma)

    # see https://pysindy.readthedocs.io/en/latest/_modules/pysindy/optimizers/constrained_sr3.html
    # NOTE: as of 6 Nov 2025, constrainedsr3 has a silent dependency on cvxpy.
    sindy_opt = pysindy.ConstrainedSR3(
        #threshold=1e-4,
        relax_coeff_nu=1e-3,
        max_iter=10000,
        constraint_lhs=constr_lhs,
        constraint_rhs=constr_rhs,
        equality_constraints=True,
        regularizer='l1',
        constraint_order="target", # dx/dt terms, then dy/dt terms, then dz/dt terms.
        tol=1e-6,
        initial_guess=W_0
    )
    
    library = pysindy.PolynomialLibrary(degree=maxdeg)

    model = pysindy.SINDy(
        optimizer=sindy_opt,
        feature_library=library
    )

    model.fit(X, t=t)
    W_approx = model.coefficients()
    return W_approx

def create_performance_profile(W_exact, W_approx, initial_cond, t_span=[0,10], nzthresh=1, varnames=None,
    fig=None,ax=None, fontsize=12):
    '''
    produce card illustrating results; reconstructed timeseries and coefficient matrices. 
    W_exact : coefficient matrix for ground truth
    W_approx : discovered/fit
    initial_cond : initial condition for forward simulations to plot
    t_span : [tmin, tmax] for ODE solver.
    nzthresh : integer, number of digits past decimal for threshold for displaying coeff values.
        Note we choose threshold of 0.5*10**(-nzthresh) for the conditional.
        Default: 1 (i.e. 1 digit past decimal)
    varnames : list of strings to use for variable names to label matrices.
        By default [x_0, x_1, ...] are used.
        
        NOTE: all variable names are TeXified (surrounded by dollar signs and rendered). 
        You shouldn't put these in yourself.
    '''
    from matplotlib import pyplot as plt
    # TODO: best way to incorporate maxdegree d? I have it hard coded.
    #maxdegree=2
    n,sigma = np.shape(W_exact)
    # ew, this isn't the best way
    for i in range(0,10):
        if _poly_exp_basis_func_count(n,i) == sigma:
            maxdegree=i
            break
    
    
    if varnames is None:
        varnames = [f'x_{i}' for i in range(n)]
    assert len(varnames)==n
    
    if fig is None:
        fig,ax = plt.subplot_mosaic('''
            AA
            BC
            ''', 
            constrained_layout=True,
            figsize=(10,6)
        )
    
    ##
    # build timeseries
    t = np.linspace(t_span[0], t_span[1], 400) # TODO: is hard-coding 400 ever a problem?
    
    rhs_exact = ode_rhs_generator(W_exact, maxdegree=maxdegree)
    resol_exact = sp.integrate.solve_ivp(rhs_exact, t_span, initial_cond, t_eval=t)
    
    ax['A'].set_prop_cycle(plt.cycler(color=[plt.cm.tab20(2*j) for j in range(10)]))
    ax['A'].plot(resol_exact.t, resol_exact.y.T, lw=2, label='Exact')
    
    rhs_approx = ode_rhs_generator(W_approx, maxdegree=maxdegree)
    resol_approx = sp.integrate.solve_ivp(rhs_approx, t_span, initial_cond, t_eval=t)
    
    ax['A'].set_prop_cycle(plt.cycler(color=[plt.cm.tab20(2*j+1) for j in range(10)]))
    ax['A'].plot(resol_approx.t, resol_approx.y.T, lw=6, ls='--',  label='Reconstructed')
    
    #ax['A'].set(xlabel=r'$t$', ylabel='State')
    ax['A'].set_xlabel(r'$t$', fontsize=fontsize)
    ax['A'].set_ylabel('State', fontsize=fontsize)
    
    ##
    # put in the coefficient matrices
    minc,maxc = -abs(W_exact).max(), abs(W_exact).max()
    
    ax['B'].matshow(W_exact, vmin=minc, vmax=maxc, cmap=plt.cm.bwr, alpha=0.5)
    ax['C'].matshow(W_approx, vmin=minc, vmax=maxc, cmap=plt.cm.bwr, alpha=0.5)
    
    ##
    # Conditionally label cell with their values
    # modified from https://stackoverflow.com/a/20998634
    mat_fstring = '{' + f':0.{nzthresh}f' + '}'
    thresh = 0.5*10**(-nzthresh)
    
    for (i, j), z in np.ndenumerate(W_exact):
        if abs(z) > thresh:
            ax['B'].text(j, i, mat_fstring.format(z), ha='center', va='center', color='k')
    for (i, j), z in np.ndenumerate(W_approx):
        if abs(z) > thresh:
            ax['C'].text(j, i, mat_fstring.format(z), ha='center', va='center', color='k')
    #
    ax['A'].set_title('Reconstructed trajectories', loc='left', fontsize=fontsize)
    ax['B'].set_title(f'Exact coefficients', loc='left', fontsize=fontsize)
    ax['C'].set_title(f'Estimated coefficients', loc='left', fontsize=fontsize)
    
    ##
    # automagically build the basis functions in latex
    powers = _power_array(n, maxdegree=maxdegree)
    phi_basis = []
    for i in range(sigma):
        term=r'$'
        for k in range(n):
            if powers[i,k]==0:
                continue
            pow=f'' if powers[i,k]==1 else f'^{powers[i,k]}'
            term += f'{varnames[k]}{pow}'
        term += r'$'
        
        # TODO: necessary?
        if i==0:
            term=r'$1$'
        phi_basis.append(term)
    #
    
    ## 
    # labels with derivative functions
    derivs = [f'$d{s}/dt$' for s in varnames]
    for panel in ['B', 'C']:
        ax[panel].set(
            xticks=range(sigma), xticklabels=phi_basis,
            yticks=range(n), yticklabels=derivs
        )
    #

    return fig,ax
#

def errors(W_exact, W_approx, t, initial_cond, maxdegree=2):
    '''
    Calculates a few measures of success and outputs a dictionary of floats.
    Inputs:
        W_exact, W_approx: two 2D arrays; coefficient matrices for ODE
        t : array; sample times for evaluation of residuals.
        initial_cond : array; initial condition for forward simulation.
        maxdegree : integer, max polynomial degree in standard basis (default: 2)
    
    Outputs:
    "errors", in the sense of comparing W_approx to W_exact:
        mae: max absolute error; i.e. max absolute error between matrix entries
        sigmamax: 2-norm of difference; or, max singular value of error matrix.
        maxviolation: infinity norm of sum of W_approx by columns
    
    "residuals", in the sense of comparing y to yhat=F(x;W), F the solution 
    operator. In this context, simulating the ODEs.
    
        forward_mae: max absolute error (forward error) across all states.
        forward_rmse: root mean square error
        forward_mre: max relative error across all states; defined as
            max( abs(x_approx - x_exact)/(eps + abs(x_exact)) ); eps=1e-14.
            This is not robust.
        
        singularity: integer; if 1, then solve_ivp halted early, most likely
            due to a singularity in the associated ODE.
    '''
    
    ### errors
    _dict = {}
    _dict['mae'] = abs(W_approx-W_exact).max()
    _dict['sigmamax'] = np.linalg.norm(W_exact-W_approx, 2)
    _dict['maxviolation'] = np.max( abs(np.sum(W_approx,axis=0)) )
    
    ### residuals
    # simulate 
    rhs_exact = ode_rhs_generator(W_exact, maxdegree=maxdegree)
    sol_exact = sp.integrate.solve_ivp(rhs_exact, t[[0,-1]], initial_cond, t_eval = t,
        atol=1e-3, rtol=1e-3)
    x_exact = sol_exact.y
    
    rhs_approx = ode_rhs_generator(W_approx, maxdegree=maxdegree)
    sol_approx = sp.integrate.solve_ivp(rhs_approx, t[[0,-1]], initial_cond, t_eval = t,
        atol=1e-3, rtol=1e-3)
    x_approx = sol_approx.y
    
    if x_approx.shape[1] < x_exact.shape[1]:
        # solver likely halted early because of a singularity in the ODE.
        # append with nans and add a flag.
        _dict['singularity'] = 1
        x_approx2 = np.nan*np.zeros(x_exact.shape)
        x_approx2[:, :x_approx.shape[1]] = x_approx
        x_approx = x_approx2
    
    # evaluate
    try:
        _dict['forward_mae'] = np.nanmax( abs(x_approx - x_exact) )
        _dict['forward_rmse'] = np.sqrt(np.nansum( (x_approx - x_exact)**2 / np.nanprod(x_exact.shape) ))
        
        _eps = 1e-14 # regularize the relative error close to double precision \eps_mach
        _dict['forward_mre'] = np.nanmax( abs(x_approx - x_exact)/(_eps + abs(x_exact)) )
    except:
        import pdb
        pdb.set_trace()
    return _dict

if __name__=="__main__":
    if False:
        #import SIR_sys
        from matplotlib import pyplot as plt
        
        # SIR
        beta = 0.3
        gamma = 0.1
        t_min = 0
        t_max = 100
        dt = 0.05
        t = np.arange(t_min, t_max+dt, dt)

        # initial conditions (make sure S+I+R=1)
        Sinit = 0.9
        Iinit = 1 - Sinit
        Rinit = 0
        xinit = [Sinit, Iinit, Rinit]
        
        ###########
        # HERE'S THE MAGIC # weow
        # set up coefficient matrix
        M = np.zeros((3,10))
        M[0,5] = -beta
        M[1,5] = -M[0,5]
        M[1,2] = -gamma
        M[2,2] = -M[1,2]
        
        # simple oscillator
        #R[0,2] = -1
        #R[1,1] = 1
        
        #
        # Generate RHS function
        rhs = ode_rhs_generator(M)
        
        print('SIR system')
        print_ode(M, varnames='SIR')
        
        # Solve system
        sol = sp.integrate.solve_ivp(rhs, t[[0,-1]], xinit, t_eval=t)
        
        
        
        ### Plot true solution with learned solution
        # Get learned coeffs
        import pytorch_ex04 as ptex
        
        M_learned = ptex.moo.m1.weight.detach().numpy()
        
        # Generate RHS
        rhs_learned = ode_rhs_generator(M_learned)
        
        # Solve system
        sol_learned = sp.integrate.solve_ivp(rhs_learned, t[[0,-1]], xinit, t_eval=t)
        
        
        #Plot S,I,R and learned solutions
        #S,I,R = x # Solution variables by row
        fig,ax = plt.subplots()
        ax.plot(sol.t, sol.y.T, label=['S', 'I', 'R'])
        ax.plot(sol_learned.t, sol_learned.y.T, label=['S learned', 'I learned', 'R learned'])
        
        ax.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('populations')
        fig.show()
    #
    
    import numpy as np
    A = np.random.normal(0,1,(5,100))
    b = np.random.normal(0,1,5)
    x = stlsq(A,b, thresh=1e-2)
    
