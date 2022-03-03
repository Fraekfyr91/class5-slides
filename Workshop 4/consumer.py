import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
class consumer():
    '''Simple class to find and print optimum'''
    def __init__(self,alpha=0.5, p1=1, p2=2, I =10):
        '''Initializer'''
        self.alpha = alpha      
        self.p1 = p1
        self.p2 = p2
        self.I = I
        

                    
        self.x1 = np.nan # not-a-number
        self.x2 = np.nan
        self.N = 100
        self.x2_max = I/p2
        
    def u_func(self,x1,x2):
        return x1**self.alpha*x2**(1-self.alpha)
    
    def value_of_choice(self,x1):
        ''' Finds optimal value of x2
        Args:
            x1 (float): Quantity of good 1.
            alpha (float): Output elasticities of good 1 and 2.
            u (float):  Utility of good 1 and 2.
            I (float/int): Income.
            p1 (float/int): Price of good 1.
            p2 (float/int): Price of good 2.
        Returns:
            Utility (float): Negative utility of good 1 and 2
        '''
        #find x2 
        x2 = (self.I-self.p1*x1)/self.p2   
        return -self.u_func(x1,x2)

    def print_solution(self,x1,x2,u):
        ''' Print function
        Args:
            X1 (float): Quantity of good 1.
            x2 (float): Quantity of good 1.
            u (float):  Utility of good 1 and 2.
            I (float/int): Income.
            p1 (float/int): Price of good 1.
            p2 (float/int): Price of good 2.
        Returns:
            prints the solutions to the consumer problem.
        '''
        print(f'x1 = {x1:.8f}')
        print(f'x2 = {x2:.8f}')
        print(f'u  = {u:.8f}')
        print(f'I-p1*x1-p2*x2 = {self.I-self.p1*x1-self.p2*x2:.8f}') 
    
    
    def solve(self, print_sol = True):

        #value_of_choice is the function we minimize
        #method is the algorithm used
        #bounds is set to use between no income and all income on good 1
        # - bounds is the space we look for the solution 
        # args is the constant parameteres in the problem

        sol_case = optimize.minimize_scalar(
            self.value_of_choice,method='bounded',
            bounds=(0,self.I/self.p1))
    
        if print_sol:
            x1 = sol_case.x
            x2 = (self.I-self.p1*x1)/self.p2
            u = self.u_func(x1,x2)
            self.print_solution(x1,x2,u)
        else: pass
        
        self.x1 = sol_case.x
        self.x2 = (self.I-self.p1*x1)/self.p2
        self.u = self.u_func(x1,x2)
    def plot(self):
                
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(1,1,1)
        # allocate memory
        x1_vecs = []
        x2_vecs = []
        us = []
        
        for fac in [0.75,1,1.25]:
            
            # fac = 1 -> indifference curve through optimum
            # fac < 1 -> ... below optimum
            # fac > 1 -> ... above optimum
                
            # a. utility in (fac*x1,fac*x2)
            u = self.u_func(fac*self.x1,fac*self.x2)
            
            # b. allocate numpy arrays
            x1_vec = np.empty(self.N)
            x2_vec = np.linspace(1e-8,self.x2_max,self.N)

            # c. loop through x2 and find x1
            for i,x2 in enumerate(x2_vec):

                # local function given value of u and x2
                def objective(x1):
                    return self.u_func(x1,x2)-u
            
                sol = optimize.root(objective, 0)
                x1_vec[i] = sol.x[0]
            
            # d. save
            x1_vecs.append(x1_vec)
            x2_vecs.append(x2_vec)
            us.append(u)
        for x1_vec,x2_vec,u in zip(x1_vecs,x2_vecs,us):
            self.ax.plot(x1_vec,x2_vec,label=f'$u = {u:.2f}$')
        
        
        self.ax.set_xlabel('$x_1$')
        self.ax.set_ylabel('$x_2$')
                
        self.ax.set_xlim([0,self.x2_max])
        self.ax.set_ylim([0,self.x2_max])

        self.ax.grid(ls='--',lw=1)
        self.ax.legend(loc='upper right')

        self.ax.plot(self.x1,self.x2,'ro') # a dot
        self.ax.text(self.x1*1.03,self.x2*1.03,f'$u^{{max}} = {self.u:.2f}$')

        x = [0,0,self.I/self.p1] # x-cordinates in triangle
        y = [0,self.I/self.p2,0] # y-cordinates in triangle
        
        # fill plot budgetline
        self.ax.plot(x,y,"--", lw=0.5, color="black", alpha=0.3) # alpha controls transparance
        