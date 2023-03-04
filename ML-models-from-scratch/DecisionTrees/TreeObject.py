import numpy as np

from BooleanFunctions import *
from Metrics import IGOptimizer

'''
DNode class, represents one node of decision tree.
'''

class DNode():
    def __init__(self, X, Y):
        self.y, self.x = Y, X # data
        self.L, self.R = None, None  # branched nodes
        self.IG, self.x0 = None, None # Ig, x0 split values
        self.cond, self.isLeaf = None, True   # condition (for visualization), isLeaf represents end of tree
        self.depth = 1
        self.msg = ""
        self.type = ''
        # condition function
        self.f = None

    def grow(self, SB, SL, MD, IGL):
        self.er = ''
        # not enough samples to branch
        if len(self.y) < SL: 
            self.msg += 'leaf'
            return False 
        # One class in the node, refuse branching
        if len(np.unique(self.y)) == 1: 
            self.msg += 'class'            
            return False

        if self.depth >= MD:
            self.msg += 'depth'            
            return False

        xsplits, igs = IGOptimizer(self.x, self.y) # optimize IG
        # find the best split, save params
        idx = np.argmax(igs) 
        self.x0 = xsplits[idx]
        self.i0 = idx
        self.IG = igs[idx]

        # not enough IG, refuse branching
        if self.IG <= IGL: 
            self.msg += f'IG < {0}'                        
            return False

        # split data
        b = self.x[:, idx] < self.x0
        nb = np.logical_not(b)
        X1, X2, Y1, Y2 = self.x[b], self.x[nb],  self.y[b], self.y[nb]

        # not enough samples to split
        if X1.shape[0] < SB or X2.shape[0] < SB:
            self.msg += 'split'                                    
            return False

        # Before splitting, it's necessary to arrange branches coreclty.
        # Since information gain doesn't consider class labels, only
        # entropy minimization goal, we should assign splitted data to 
        # correct branch (left or right) separately in respect to classes ratio.
        # Let agree, that Left branch always contain '0' class.
        # Then, consider the ratio between classes in Y1, Y2:

        #Note: function f: X-Array => Bool-Array always returns '0' classes prediction
        if np.sum(Y1==0)/ max(np.sum(Y1==1), 0.5) > np.sum(Y2==0)/max(np.sum(Y2==1), 0.5):
            self.cond = f"x{idx} < {round(self.x0, 3)}"
            self.f = BoolFunction(idx, self.x0, False)
            # create Nodes, left node shold store more '0' class labels
            self.L = DNode(X1, Y1)
            self.R = DNode(X2, Y2)
            self.R.type = 'right'
            self.L.type = 'left'
        else:
            self.cond = f"x{idx} >= {round(self.x0, 3)}"
            self.f = BoolFunction(idx, self.x0, True)
            # create Nodes, left node shold store more '0' class labels
            self.R = DNode(X1, Y1)
            self.L = DNode(X2, Y2)            
            self.R.type = 'right'
            self.L.type = 'left'

        # increase depth
        self.R.depth = self.L.depth = self.depth + 1
        # now, it's a branch, not leaf
        self.isLeaf=False
        # successfully splitted
        return True

