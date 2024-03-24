'''
Implementation of K-Means clustering using torch and numpy
'''

import numpy as np
import torch
import torch.nn as nn

import numpy as np
import torch.nn as nn

'''
Class Kmeans
Call with: 
    n_cluster: Number of clusters (int)
    max_iter: Maximum number of iterations before stopping (int)
    distance: Distance measure (str, 'euclidean' or 'manhatten' possible)
        --> If wrong parameter used, euclidean choosen by default
    device: Device to work on (str)
        --> Make sure, device exists on your hardware
    loss: Loss function to apply (nn.Module)
'''
class Kmeans:
    def __init__(self, 
                 n_clusters: int = 8,
                 max_iter: int = 300, 
                 distance: str = 'euclidean', 
                 device: str = 'cpu',
                 loss= nn.MSELoss()):
        self.n_clusters = n_clusters 
        self.max_iter = max_iter
        self.loss = loss
        self.device = device
        if(distance == 'euclidean'):
            self.calculate_distance = self.euclidean_distance
        elif(distance =='manhatten'):
            self.calculate_distance = self.manhatten_distance
        else: 
            self.calculate_distance = self.euclidean_distance
         
    '''
    Function which initializes cluster centers
    Call with num_classes: number of clusters you would like to have (int)
    Returns tensor of zeros with length num_classes
    '''
    def initialize_centers(self, num_classes):
        self.centroids = torch.stack([torch.zeros(num_classes) for i in range(self.n_clusters)]).to(self.device)
        
    '''
    Calculation of euclidean distance between two points in n-dimensional room
    '''
    def euclidean_distance(self, center, point):
        dist = torch.tensor(0, dtype = torch.float32).to(self.device)
        for i, elem in enumerate(center):
            dist += (elem - point[i])**2
        return(torch.sqrt(dist))
    
    '''
    Calculation of manhatten distance between two points in n-dimensional room
    '''
    def manhatten_distance(self, center, point):
        dist = torch.tensor(0, dtype = torch.float32).to(self.device)
        for i, elem in enumerate(center):
            dist += np.abs((elem - point[i]))
        return(dist)
        
        
    '''
    Function which assigns tensor of points to clusters
    Call with tensor of points
    Returns: 
        Cluster for every point
        Center every data point belongs to
    '''
    def assign_cluster(self, X):
        clusters = []
        Centers = torch.stack([torch.zeros(len(self.centroids[0])) for i in range(len(X[0]))]).to(self.device)
        for i, point in enumerate(X):
            dists = torch.zeros(len(self.centroids), dtype = torch.float32)
            for i, centroid in enumerate(self.centroids): 
                dist =self.calculate_distance(point, centroid)
                dists[i] = dist
            cluster = torch.argmin(dists)
            clusters.append(cluster)
            Centers[i] = torch.tensor(self.centroids[cluster])
        return(clusters, Centers)    
    
    '''
    Function which calculates distance between points and corresponding cluster centers
    Call with list of points and corresponding cluster center
    Returns average distance between points and corresponding centroids
    '''
    def calculate_dist(self, X_clustered):
        loss = torch.tensor(0, dtype = torch.float32).to(self.device)
        for point in X_clustered: 
            loss += self.calculate_distance(point[0], self.centroids[point[1]])
        loss /= len(X_clustered)
        return(loss)
        
    '''
    Function which fits KMeans
    First initializes cluster centers based on distance between points
    Adapts cluster centers iteratively afterwards
    Call with: 
        X_train: Data to fit model on (torch.Tensor)
        stopping_delta: Minimum difference for early stopping (float)
    '''
    def fit(self, 
            X_train, 
            stopping_delta: float = 0.0):
        idx = np.random.choice(range(len(X_train)))
        self.centroids = [X_train[idx]]
        
        for i in range (self.n_clusters - 1):
            dists = torch.stack([torch.sum(torch.stack([self.calculate_distance(x, centroid) for centroid in self.centroids]))for x in X_train])
            idx = torch.argmax(dists)
            self.centroids.append(X_train[idx])
        iteration = 0
        centroids_previous = None
        while(iteration < self.max_iter):
            clusters, centers = self.assign_cluster(X_train)
            means = []
            for i in range(self.n_clusters):
                mask = torch.tensor(clusters) == i                
                points = torch.stack(list(X_train))[mask]
                means.append(torch.mean(points, dim = 0))
                
                
            centroids_previous = self.centroids
            self.centroids = torch.stack(means)
            for i, centroid in enumerate (self.centroids):
                if(torch.isnan(centroid).any()):
                    self.centroids[i] = centroids_previous[i].detach()
            if centroids_previous is not None:
                deltas = torch.norm(self.centroids - centroids_previous, dim=1)
                if torch.all(deltas < stopping_delta):
                    print(f'Converged after {iteration}')
                    break
            
            iteration += 1            
            
