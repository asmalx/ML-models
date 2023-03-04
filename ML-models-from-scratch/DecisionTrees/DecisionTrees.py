import numpy as np

from TreeObject import DNode
from TreeVisualizer import DrawDTree
from BooleanFunctions import MergeFunctions, NotF, FNothing


class DecisionTreeClassifier():
    def __init__(self, max_depth=None, 
                 min_sample_split = 1,
                 min_leaf_samples = 1,
                 min_IG_split=1e-10, verbose=False):
        self.max_depth = max_depth if not max_depth is None else 1e100
        self.min_sample_split = min_sample_split
        self.min_leaf_samples = min_leaf_samples
        self.min_IG_split = min_IG_split
        self.verbose = verbose

        
    def fit(self,X, Y):
        # create root
        self.root = DNode(X, Y)
        # build tree
        self.print_msg("Building tree...")
        self.__grow_tree(self.root)        
        self.print_msg("Merging nodes functions...")        
        # Get rules (merge functions of nodes)
        self.F, self.list_rules = self.__traverse_merge_conditions(self.root)
        self.nleaves = len(self.list_rules)
        self.print_msg("Done.")

        pass
    
    def predict(self,X):
        return 1 - self.F(X).astype('uint8')
    
    def get_depth(self):
        return self.__get_depth_rec(self.root)
    
    def get_leaves_count(self):
        return self.nleaves
    
    def get_rules(self,merge_to_one=True):
        if merge_to_one: return self.F
        return self.list_rules
    
    def score(self,X, Y):
        return np.sum(self.predict(X)==Y) / len(Y)
    
    def visualize(self, save=False, name=None):
        self.print_msg("Drawing tree...")
        visualizer = DrawDTree(self.root)
        arr = visualizer.get_image()
        if save: visualizer.save_jpg(name)
        self.print_msg("Done.")
        return arr

    def print_msg(self, msg):
        if self.verbose:
            print(msg)
    
    def __grow_tree(self, root):
        stack = [root]
        while len(stack):
            node = stack.pop()
            node.grow(self.min_sample_split, self.min_leaf_samples, self.max_depth, self.min_IG_split)
            if not node.L is None: 
                stack.append(node.L)
            if not node.R is None: 
                stack.append(node.R)    
    
    
    def __traverse_merge_conditions(self, root):    
        stack = [(root, FNothing())] 
        leaf_functs = []
        for idx, (node, g) in enumerate(stack):
            if not node.L is None: 
                stack.append((node.L, MergeFunctions(g, node.f, np.logical_and)))    
            if not node.R is None: 
                stack.append((node.R, MergeFunctions(g, NotF(node.f), np.logical_and)))

            if node.R is None and node.L is None and node.type=='left':
                leaf_functs.append(g)

        f = leaf_functs[0]
        for g in leaf_functs: f = MergeFunctions(f,g, np.logical_or)
        return f, leaf_functs

        
    def __get_depth_rec(self, root, depth=0):        
        # calculate tree depth recursively
        if root is None: return depth
        return max(self.__get_depth_rec(root.L, root.depth), self.__get_depth_rec(root.R, root.depth))

    