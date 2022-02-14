
   
"""
Defines K Nearest Neighbor machine learning classification
structure. Has train and predict methods that allow data to 
be appropriately grouped and classified from.
"""


import numpy as np
from src.distance import euclidean


class KNearestNeighbors:


    def __init__(self, k=3, distance_metric=euclidean):
        """ Initialize k value and distance metric used for model. """
        self.k = k
        self.distance = distance_metric
        self.data = None


    def train(self, P, q):
        """ Zip labels and input data together for classification. """
        # value error raised for inappropriate data type inputs
        if len(P) != len(q) or type(P) != type(q):
            raise ValueError("Innapropriate data types. ")
        # convert ndarrays to lists
        if type(P) == np.ndarray:
            P, q = P.tolist(), q.tolist()
        # set data attribute containing instances and labels
        self.data = [P[i]+[q[i]] for i in range(len(P))]


    def predict(self, A):
        """ Predict class based on k-nearest neighbors. """
        neighbors = []
        # create mapping from distance to instance
        distances = {self.distance(x[:-1], A): x for x in self.data}
        # collect classes of k instances with shortest distance
        for key in sorted(distances.keys())[:self.k]:
            neighbors.append(distances[key][-1])
        # return most common vote
        return max(set(neighbors), key=neighbors.count)
