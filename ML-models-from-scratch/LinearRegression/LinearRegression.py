import numpy as np
from SimpleGDOptimizer import GradientDescent

'''
Linear Regression model.
Nothing unusual =)


1. fit method, learn the model
 - x: Input data, 2dim [n_samples, k_features]
 - y: Target data, 1dim [n_samples]
 - method:
     'equation' - analithical solution w = xx.T
     'iterative' - using SGD

2. predict method, calculates prediction
3. score method, calculates R2 score
'''


class LinearRegression():
    def __init__(self):
        self.w = None # declare coefficients
        
    def fit(self, x, y, method='equation', max_iters=1000, eps=1e-8, **kwargs):
        self.MI, self.EPS = max_iters, eps # iterative fit params
        if len(np.array(x).shape) == 1: self.X = np.array(x, ndmin=2).T # if x is 1-dimensional - convert
        else: self.X = x 
        self.n, k = self.X.shape # obtain data shape
        # add one extra dim for bias (intercept term)
        self.X = np.concatenate([self.X, np.ones((self.n,1))], axis=-1)
        self.w = np.ones((k+1, 1)) # initialize coefficients (including intercept)
        self.Y = y # save target data
                
        if method == 'equation': # if iterative, call iterative with specified params
            self.__fit_equation(self.X, self.Y)
            return self.coef, self.intercept
        else: # else use equation method
            self.__fit_iterative(self.X, self.Y, **kwargs)
            return self.coef, self.intercept
        
    def predict(self, x): 
        if len(np.array(x).shape) == 1: x = np.array(x, ndmin=2).T # if x is 1-dimensional - convert
        return x @ self.coef + self.intercept # calc
    
    def score(self, x, y):
        # R2 metric, 
        ss_tot = np.sum(np.square(y - np.mean(y))) # total sum of squares
        ss_res = np.sum(np.square(y - self.predict(x))) # residuals square sum
        return 1. - (ss_res / ss_tot) # metric

    
    
    def __mse(self, w):
        return np.mean(np.square(self.Y - self.X @ w), axis=0) # MSE, cost function, generally L(w, x, y)
    
    def __dmse(self, w): # derivative of MSE, in respect to w (dL / dw)
        return - self.X.T @ (self.Y - self.X @ w) / self.n
    
    
    def __fit_equation(self, x, y):
        try: # calc inverse matrix directly,
            self.w = np.linalg.inv(x.T @ x) @ (x.T @ y)
        except: #  if error - use iterative, approximation alg
            self.w = np.linalg.pinv(x.T @ x) @ (x.T @ y)

        self.coef = self.w[:-1:, ]
        self.intercept = self.w[-1, ]
    
    def __fit_iterative(self, x, y, **kwargs): # apply gradient descent, defined before
        opt = GradientDescent(**kwargs) # use specified params
        opt.build(F=self.__mse, x_shape=self.w.shape, f=self.__dmse, x0=self.w) # build
        opt.optimize(self.MI, self.EPS) 
        self.w = opt.x0 # get min points value
        self.coef = self.w[:-1] # split into two coefs
        self.intercept = self.w[-1]
       