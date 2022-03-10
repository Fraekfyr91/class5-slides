import numpy as np

from scipy import optimize
class problem():
    '''Simple class to find and print optimum'''
    def __init__(self,N=10000, p = [1.4,1,1], k= 2, kappa = 0.5, eps = 1e-8, maxiter = 500, disp = True):
        '''Initializer'''
        self.N = N      
        self.J = 3
        self.Sigma_lower = np.array([[1, 0, 0], [0.5, 1, 0], [0.25, -0.5, 1]])
        self.Sigma_upper = self.Sigma_lower.T
        self.Sigma = self.Sigma_upper@self.Sigma_lower
        self.alphas = np.exp(np.random.multivariate_normal(np.zeros(self.J), self.Sigma, self.N))
        self.p = p
        self.kappa = kappa
        self.eps = eps
        self.maxiter = maxiter
        self.disp = disp
        self.k = k
        
    def demand_good_1_func(self,p):
        ''' Defines income and the demand for good 1
        Args:
            p (list): Prices, has to be equal to J.
            k (float): endowment of good 1.

        Returns:
            Demand for good 1
        '''
        
        assert len(self.p) == self.J # number of prices must be equal to the number of goods
        
        self.I = [self.p[0] * self.k] + self.p[1:] # Income equal endowment times prices
        
        return self.alphas*self.I/self.p[0]

    
    def excess_demand_good_1_func(self,p):
        ''' Find the excess demand for good 1
        Args:
            p (list): Prices, has to be equal to J.
            k (float): endowment of good 1.
            alphas (numpy.ndarray): Preference parameter, hard coded Sigma.
        Returns:
            Excess demand for good 1
        '''
   
        # a. demand
        demand = np.sum(self.demand_good_1_func(self))

        # b. supply
        supply = self.k * self.alphas.size
        

        # c. excess demand
        excess_demand = demand-supply
    
        return excess_demand
    
    def find_equilibrium(self):
        ''' Finds equilibrium
        Args:
            p (list): Prices, has to be equal to J.
            k (float): endowment of good 1.
            alphas (numpy.ndarray): Preference parameter, hard coded Sigma.
            eps (float): tolerance level of precision
            maxiter (int): the maximum iterations to find equilibrium
            kappa (float): update parameter
        Returns:
            A list of equilibrium prices
        '''
        t = 0
        while True:
            # a. step 1: excess demand
            Z1 = self.excess_demand_good_1_func(self)

            # b: step 2: stop?
            if  np.abs(Z1) < self.eps or t >= self.maxiter:
                if self.disp:
                    print(f'{t:3d}: p1 = {self.p[0]:12.8f} -> excess demand -> {Z1:3.8f}')
                    break    
                else: break
            # c. step 3: update p1
            self.p[0] = self.p[0] + self.kappa*Z1/self.alphas.size

            # d. step 4: return 
            if t < 5 or t%25 == 0:
                if self.disp:
                    print(f'{t:3d}: p1 = {self.p[0]:12.8f} -> excess demand -> {Z1:3.8f}')
                else: pass
            elif t == 5:
                if self.disp:
                    print('   ...')
    
                else: pass
            t += 1    

        return self.p