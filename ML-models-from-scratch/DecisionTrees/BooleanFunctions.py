from copy import deepcopy
import numpy as np

'''
Custom boolean function classes,
 - allows safe nested functions with deepcopy
 - custom __repl__ method, for convenient investigation
     of tree rulues

'''

# Bool function is used for tree node, 
# feature_index - index of deature in X, X[:, feature_index]
# threshold - value to compare, feature > threshold
# sign_gt - sign, '>=' or '<'
class BoolFunction():
    def __init__(self, feature_index, threshold, sign_gt=True):
        self.a = float(threshold)
        self.i = int(feature_index)
        self.sign = bool(sign_gt)
    
    def __call__(self,X):
        if self.sign: return (X[:, self.i] >= self.a)
        return (X[:, self.i] < self.a)

    def __str__(self):
        sign_name = ''
        if  self.sign : sign_name = ' >= '
        else: sign_name = ' < '        
        return f"x{self.i}" + sign_name + str(round(self.a, 3))

# Merge two bool functions, equivalent to lambda x: mapf(f(x), g(x))
class MergeFunctions():
    def __init__(self, f, g, mapf):
        self.f = deepcopy(f)
        self.g = deepcopy(g)
        self.map = deepcopy(mapf)
    
    def __str__(self):
        map_name = ''
        if 'and' in str(self.map): map_name = ' && '
        if 'or' in str(self.map): map_name = ' || '            
        
        return str(self.f) + map_name + str(self.g)
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, X):
        return self.map(self.f(X), self.g(X))

# Negation of function, equivalent to lambda x: np.logical_not(f(x))  
class NotF():
    def __init__(self, f):
        self.f = deepcopy(f)
        
    def __call__(self, X):
        return np.logical_not(self.f(X))
    
    def __str__(self):        
        return  '!(' + str(self.f) + ')'

# Identity function
class FNothing():
    def __init__(self):
        pass
        
    def __call__(self, X):
        return np.ones(X.shape[0]).astype('bool')
    
    def __str__(self):        
        return  'True'
    