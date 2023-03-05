import numpy as np


'''
GD method. Applies Gradient Descent to a target function.

1. Constructor params:
 - learning_rate: learning rate (float)
 - momentum: momemtum (bool)
 - nesterov: using nesterov momentum (bool)
 - beta: momentum param (float 0-1)

2. build method, allows to appoint x data shape, target F and/or inital x
 - F: target function to minimize (func(x))
 - x_shape: input data shape (tuple)
 - f: derivative of F, if None - using numerical dF calculation
 - x0: inital vector, if None - using zeros

3. calculate_gradients method, returns gradients of F at current point x
 - xi: point, at which gradients is calculated; if None - use current x0
 
4. do_step method, make one GD step
 - grads: gradients of F at current point xi
 - xi: point, at which calc is going
 
 
5. optimize method, applies gd to target with specified params
 - max_iter: max iteration count
 - epsilon: defines convergence condition
'''




class GradientDescent():
    def __init__(self, learning_rate=1e-3, momentum=False, nesterov=False, beta=0.5, noised_grads=(0., 0.)):
        self.lr = learning_rate
        self.grads_prev1 = None
        self.grads_prev2 = None
        self.builded = False
        self.beta = 1. - beta
        self.stat = []
        self.iters_passed = None
        self.noised_grads = noised_grads
        
        if nesterov: self.apply_grads_f = self.__apply_grads_nmomentum 
        elif momentum and not nesterov: self.apply_grads_f = self.__apply_grads_momentum 
        elif not momentum and not nesterov: self.apply_grads_f = self.__apply_grads_bare
        else: self.apply_grads_f = self.__apply_grads_bare
    
    def build(self, F, x_shape, f=None, x0=None):
        self.xshape = x_shape
        self.Kk = x_shape[0]
        self.F = F
        if x0 is None: x0 = np.zeros(self.xshape)
        self.x0 = x0
        if f is None: f = self.__central_df
        self.calculate_gradients = f
        
        self.grads_prev1 = self.x0
        self.grads_prev2 = self.x0
        self.eps = 1e-3
        
        self.Meps = np.eye(self.Kk) * self.eps
            
        self.builded = True
        
    def calculate_gradients(self, xi=None):
        if xi is None: xi = self.x0
        return self.f(xi)
    
    def __apply_gradients(self, xi):
        return self.apply_grads_f(self.calculate_gradients(xi), xi)

    def do_step(self, xi, grads):
        return xi - self.lr*self.apply_grads_f(grads, xi)
    
    def optimize(self, max_iter=1e5, epsilon=1e-8, store_steps=False):        
        self.stat = []
        for i in range(int(max_iter)):
            self.iters_passed = i
            x00 = self.x0
            self.x0 = self.__apply_gradients(self.x0)
            if (np.max(np.abs(x00 - self.x0)) < epsilon and i > 1): return
            if store_steps: self.stat.append(self.x0)
            
    
    # bare GD alg
    def __apply_grads_bare(self, grads, xi): 
        return xi - self.lr * grads
    
    # default momentum
    def __apply_grads_momentum(self, grads, xi):
        new_grads = ( self.beta * self.grads_prev1 + (1. -  self.beta) * grads)
        self.grads_prev1 = new_grads
        return xi - self.lr * new_grads
    
    # Nesterov momentum
    def __apply_grads_nmomentum(self, grads, xi):
        new_grads = ( self.beta * self.grads_prev2 + (1. -  self.beta) * grads)
        self.grads_prev2 = self.grads_prev1
        self.grads_prev1 = new_grads
        return xi - self.lr * new_grads
    
    
    # simple numerical derivative calculation
    def __central_df(self, x):
        df = 0.5*(self.F(np.repeat(x, self.Kk, axis=-1) + self.Meps) - self.F(np.repeat(x, self.Kk, axis=-1) - self.Meps)) / self.eps
        return df.reshape(self.xshape) + np.random.normal(*self.noised_grads, self.xshape)
    
  