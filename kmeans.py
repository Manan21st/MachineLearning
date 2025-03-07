import numpy as np
from enum import Enum

class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"

class KMeans:
    def __init__(self,k=3,iterations=100, formula=DistanceMetric.EUCLIDEAN):
        self.k = k
        self.iterations = iterations
        self.formula = formula
        self.clusters = None

    def distance(self, x1, x2):
        if self.formula == DistanceMetric.EUCLIDEAN:
            return np.sqrt(np.sum((x1-x2)**2)) # Euclidean distance
        elif self.formula == DistanceMetric.MANHATTAN:
            return np.sum(np.abs(x1 - x2)) # Manhattan distance
        elif self.formula == DistanceMetric.COSINE:
            return 1 - (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))) # Cosine distance
        else:
            raise ValueError("Invalid formula")

    def random_centroids(self, X):
        np.random.seed(23)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        clusters = {i: {'center': X[idx], 'points': []} for i, idx in enumerate(indices)}
        return clusters

    def assign_cluster(self, X, clusters):
        for cluster in clusters.values(): 
            cluster['points'] = []  

        for idx in range(X.shape[0]):
            distances = [self.distance(X[idx],clusters[i]['center']) for i in range(self.k)]            
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx]['points'].append(X[idx])
        return clusters
    
    def update_cluster(self, clusters):
        for idx in range(self.k):
            points = np.array(clusters[idx]['points'])
            if points.shape[0] > 0:
                clusters[idx]['center'] = np.mean(points, axis=0)
            clusters[idx]['points'] = []
        return clusters
    
    def fit(self, X):
        self.clusters = self.random_centroids(X)
        for _ in range(self.iterations):
            self.clusters = self.assign_cluster(X, self.clusters)
            self.clusters = self.update_cluster(self.clusters)

    def predict(self, X):
        predictions = []
        for idx in range(X.shape[0]):
            distances = [self.distance(X[idx],self.clusters[i]['center']) for i in range(self.k)]
            cluster_idx = np.argmin(distances)
            predictions.append(cluster_idx)
        return predictions  
