
import torch
torch.manual_seed(0)

import tools

from matplotlib import pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, d, maxdegree=2, save_every=100, thresh=1e-3):
        super(Net, self).__init__()
        p = tools._poly_exp_basis_func_count(d, maxdegree=maxdegree)
        
        self.m1 = torch.nn.Linear(p, d, bias=False) #maps R^p to R^d.
        #self.m1.weight = torch.zeros(shape)
        #self.m1.retain_grad()
        
        # NOTE: MSELoss is a Frobenius  norm scaled by number of array elements (if "mean" reduction)
        self.loss = torch.nn.MSELoss(reduction='mean')
        
        self.thresh = thresh
        self.save_every = save_every
        self.history = {'iter':[], 'mse':[], 'l1_reg':[], 'nz_comp': []}
        
    def forward(self, x):
        return self.m1(x) # bias + c[0]*x[0] + c[1]*x[1]
    
    def train(self, X, y, niter=10001):
        if not (type(X) is torch.Tensor):
            X = torch.Tensor(X)
        if not (type(y) is torch.Tensor):
            y = torch.Tensor(y)
        
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        for i in range(niter):
            self.zero_grad()
            pred = self.forward(X)
            
            l2 = self.loss(pred, y)
            
            # only addition (?!)
            lam = 0.004
            l1_reg = lam*torch.linalg.vector_norm(self.m1.weight, 1)
            
            err = l2
            err = err + l1_reg
            
            grad = err.backward()
            #import pdb
            #pdb.set_trace()
            
            optimizer.step()
            
            po = self.m1.weight.detach().numpy().flatten()
            
            if i%self.save_every==0:
                #self.history = {'iter':[], 'mse':[], 'l1_reg':[], 'norm_grad': [], 'nz_comp': []}
                self.history['iter'].append(i)
                self.history['mse'].append(float(l2.detach()))
                self.history['l1_reg'].append(float(l1_reg.detach()))
                #self.history['norm_grad'].append(torch.linalg.norm(grad, torch.inf))
                self.history['nz_comp'].append(int( (abs(self.m1.weight) > self.thresh).sum() ))
                #self.history['norm_grad'].append(torch.linalg.vector_norm(self.m1.grad, 1))
            
            #coeff_err = self.m1.bias
            if i%(niter//10)==0:
                print(f"End of iter {i} MSE: {err:.2}" )
                nzpairs = torch.where(abs(self.m1.weight) > self.thresh)
                _count = len(nzpairs[0])
                
                print(f'\tNumber of nonzero weights: {_count}')
#

