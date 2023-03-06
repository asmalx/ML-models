# ML-models
Repo contain implementation of some ML modes.

## 1. Directories overview
### 1.1 CycleGAN

There is my implementation of cycleGAN, based predominantly on origin paper.
Implementation uses tensorflow package.
Note, that model was tested only with MNIST data, however good result was achieved.

### 1.2 Ml-models from-scrath
There are some classical ML models, implemented only with numpy and emded python utils for a kind of an educational purposes.
Folder consist of subfolders, for each model correspondingly.
Each models folder has some utils files, main file with the model actually (it has the same name, as directory) and `run.ipynb` jupyter notebook, which contain example with artificially generated data and more detailed description of methods.

#### 1.2.1 Linear regression

Implementation includes two optional solution methods: 
 - equation, viz. $W = (X^T \times X)^{-1} \times (X^T \times Y)$ 
  - iterative, using Gradient Descent (implementation of algorithm is placed to separate file). GD allows to use momentum and Nesterov momentum, calculate derivative with simple numerical method
  
There is no any specific methods, just a kind of default and concise ones:

1. `fit(x, y, method, max_iters, eps, <grad.descent. params.>)`
2. `predict(x)`
3. `score(x, y)` - calculate $R^2$ score

#### 1.2.2 K-Means algorithm
Bare implementation of classical algorithm. Uses random initialization of centroids, numpy functions are used mostly insted of loop iteration (except main loop of fit)

Methods description:

1. `__init__(clusters, norm_order, algorithm_runs, max_iter, rand_seed)` 

***norm_order*** defines order of Minkowski norm (1 -manhattan, 2 - euclidean, ...)

***algorithm_runs*** - count of algorithm runs with rand initialization, choose best result at the end.

2. `fit(x, eps)` - returns (centroids coordinates, score)

***eps*** convergence constant

3. `predict(x)` - returns cluster labels for x

4. `transform(x)` - returns array of distances to centroids, distances[n_samples, k_clusters]

4. `get_centroids(x)` - returns centroids coordinates

5. `get_inertia()` - calculates inertia (mean squared distance to closest centroid) for fitted data

6. `score(x)` - returns silhouette score [-1, 1]



#### 1.2.3 Decision trees

Bare implementation of classical algorithm, but it doesn't include cp prunning option. Uses best split always along all features. Iterative methods is used mostly, except depth calculation, which is recursive one. Implementation allows to:
1. Build tree with constraints ***min_leaf_size, min_split_size, max_depth, min_information_gain_split***
2. return depth of tree 
3. return leaves count
4. calculate score (accuracy)
5. return callable object (function f: array -> Bool) with convenient representation, which predicts 0 class
6. visualize tree
 
More details are described in file `run.ipynb`


