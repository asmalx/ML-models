import numpy as np

class KMeans():
    
    def __init__(self, clusters, distance_order=2, nruns=10, max_iter=1000, random_seed = 1):
        import numpy.random as rnd
        if not random_seed is None: rnd.seed(random_seed)
        self.rnd = rnd # random object
        self.k = clusters # clusters count
        self.runs = nruns # number of algorithm runs
        self.max_iters = max_iter 
        self.p = distance_order # order of minkowski norm (1 -manhattan, 2 - euclidean)
        self.c = None
    
    # input data x[n_samples, n_features]
    def fit(self, X, eps=1e-4):
        self.x = X # save data
        centroids = []
        scores = []
        for nrun in range(self.runs):
            self.c = self.__init_centroids() # generate centroids
            
            for it in range(self.max_iters):
                c_new = self.__get_moved_centroids() # move and get new centroids
                if np.sum(np.abs(self.c - c_new)) < eps: break # convergence condition                    
                self.c = c_new # assign new centroids
                
            centroids.append(self.c)
            scores.append(self.score(X))
        
        # best centroids
        self.c = centroids[np.argmax(scores)] 
        return self.c.copy(), np.max(scores)
    
    # returns array A[n_samples] with clusters lbl {0, 1, ..., klusters}
    def predict(self, X):
        return np.argmin(self.transform(X), axis=-1)
    
    
    # returns array A[n_samples, k_clusters] 
    # with distances to each claster 
    def transform(self, X):
        return np.apply_along_axis(lambda x: self.__norm(x, self.c), -1, X)
    
    
    def get_centroids(self):
        return self.c.copy()
    
    def get_inertia(self):
        return np.mean(np.square(np.min(self.transform(self.x), axis=-1)))
        
    # silhouette score
    def score(self, X):
        n = X.shape[0] # len of data
        clbls = self.predict(X) # get cluster labels
        
        # merge X and Cluster labels together
        xy = np.concatenate([X,clbls[:, np.newaxis]], axis=1)
        
        def score_for_xyj(xyi):
            xi, k = xyi[0:-1], xyi[-1] # get sample data, cluster which sample belongs to
            # find index of nearest to current point cluster
            m = np.argmin(self.__norm(xi, self.c)[[j for j in range(self.k) if j !=k]])       
            ai = np.mean(self.__norm(X[clbls==k], xi)) # mean distance to samples of friend cluster
            bi = np.mean(self.__norm(X[clbls==m], xi)) # mean distance to samples of closest enemy cluster
            return (bi - ai) / max(ai, bi)
        
        return np.mean(np.apply_along_axis(score_for_xyj, -1, xy))
        
        
    def __init_centroids(self):
        idx = np.arange(0, self.x.shape[0]) # indexes array
        self.rnd.shuffle(idx) # shuffle
        return self.x[idx[0:self.k]] # return random K points
    
    
    # calculate norm between sample and clusters
    # x[m_features] -> d[k_clusters]
    def __norm(self, x, c):
        return np.power(np.sum(np.power(np.abs(np.repeat(x[np.newaxis, :], \
            repeats=c.shape[0], axis=0) - c),self.p), axis=-1), 1./self.p)

    def __get_moved_centroids(self):
        # get cluster labels
        clbls = self.predict(self.x)
        # calc average coords of samples in each cluster
        return np.array([np.nan_to_num(np.mean(self.x[clbls==cl], axis=0), nan=0.) for cl in range(self.k)]) 
        
        
    
          