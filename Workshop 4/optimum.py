import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
class optimum():
    '''Simple class to find and print optimum'''
    def __init__(self,N=100, c=1.0, guess = 0):
        '''Initializer'''
        self.N= N       # Per period payoff
        self.x1_vec = np.linspace(-10,10,N)    # Discount factor
        self.x = [self.x1_vec]
        self.x_guess = guess
        self.obj = lambda x: self.f(self.x)
   
    def f(self, x):

        '''
        Defines an equation.

        Args:
        x (list): list of variable arrays

        Returns:
        eq_1 (scalar): Function value
        '''
        if len(self.x)==0: 
            raise ValueError ("No variables defined")
        elif len(self.x)==1: # to ensure input is valid - in this case only 1 allowed
            eq_1 = np.sin(x[0])+0.05*x[0]**2 
            return eq_1
        elif len(x)==2: # to ensure input is valid - in this case only 2 variables allowed
            eq_1 = np.sin(x[0])+0.05*x[1]**2 
            return eq_1
        elif len(self.x)>2: 
            raise ValueError ("Too many variables defined")
            
    def solve(self, disp = True):
        '''
        Finds the minimum value of the function f()
        
        Args:
        disp (Bool): True or False, Prints solver outcomes
        
        Returns:
        Function minimum values for x and f(x).
        '''
        
         # optimizer needs a starting point for the two values     
        obj = lambda x: self.f(x) #objective function to optimize - in this case minimize
 
        res = optimize.minimize(obj, self.x_guess,method="Nelder-Mead") #Nelder-mead is standard and simple method
        if disp:
            print("-----------")
            print(res.message)
            print("-----------")
        else: pass
        #c.unpacking results
        x1_best_scipy = res.x[0]
        f_best_scipy = res.fun
        if disp:
            # d. print
            print(f'Using numerical solver the optimal values are:')
            print(f'Function = {f_best_scipy.item():.4f};  x1 = {x1_best_scipy:.4f}')    
        else: pass
        return  [x1_best_scipy,f_best_scipy.item()]
    
    def twod_plot(self, solve):
        """
        Plots graph

        Args:
        x1_vec (numpy.ndarray): vector with values to plot

        Returns:
        eq_1 (scalar): Function value
        """
        fig = plt.figure(figsize=(10,5)) # define new figure object
        ax = fig.add_subplot(111) # add subplot
        ax.plot(self.x1_vec,self.f(self.x)) # plot 2-dimensional function
        ax.plot(solve[0], solve[1], 'ro')
        ax.text(solve[0] - 2, solve[1] + 0.5, f'Minimum is ({solve[0]:.3f}, {solve[1]:.3f})')
        #add lines
        for y in range(-1, 6):    
            plt.plot(range(-10,11), [y] * len(range(-10, 11)), "--", lw=0.5, color="black", alpha=0.3)

        ax.xaxis.label.set_fontsize(14) #set label fontsize to 14
        ax.yaxis.label.set_fontsize(14)
        ax.set(xlabel="$x_1$", ylabel = "$f(x_1)$",xlim = ([-10,10])) #set xlabel,ylabel and xlimit
        for item in ax.get_yticklabels()+ax.get_xticklabels(): # set ticklabels to fontsize 14
            item.set_fontsize(14)

        #remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
