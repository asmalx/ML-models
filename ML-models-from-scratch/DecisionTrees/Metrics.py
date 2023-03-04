import numpy as np

'''
 Calculate classes probabilities (P)
 [class1 / classes, class2 / classes, ...]
 '''
def p_classes(y):
    n = len(y)
    _, c = np.unique(y, return_counts=True)   
    return c / n

'''
 information entropy function (E)
 E = -Sum_i( Pi * log2(Pi))
 '''

def entropy(y):
    return -np.sum((p_classes(y)*np.log(p_classes(y))) / np.log(2))
 
'''
 information gain function (IG), measure of reduction of entropy
 IG for split y by y1, y2:
     IG2 = E(y) - len(y1) / len(y) * E(y1) - len(y2) / len(y) * E(y2) 

'''

def information_gain(y, y1, y2):
    return entropy(y) - (len(y1) * entropy(y1) - \
    - len(y2) * entropy(y2)) / len(y)

# wrap to convert information_gain function from IG(i) to IG(x0)
def IG_fromX(x,y, x_split):
    b = np.squeeze(x < x_split)
    return information_gain(y, y[b], y[np.logical_not(b)])


'''
IG maximizer
Receives k features x[n, k], y
Returns best index to split, IG value

NOTE: x should be numerical only

C = O(n*k)
'''


def IGOptimizer(X, y):
    # iterate by features
    x0_list, max_IG_list = [], []
    for i in range(X.shape[1]):
        x = X[:, i]
        x0 = None # inital x for condition (x < x0)
        max_IG = -1.
        # iterate by potential values to split
        for i, xi in enumerate(x):
            ig = IG_fromX(x, y, xi) # cacl IG
            if ig > max_IG:  # if more that stored, store
                max_IG = ig
                x0 = xi
        x0_list.append(x0) 
        max_IG_list.append(max_IG)
    return np.array(x0_list), np.array(max_IG_list)