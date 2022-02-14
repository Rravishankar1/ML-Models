""" 
Defines Random Forest machine learning classification
structure. Has fit predict and score methods that allow 
data to be appropriately grouped by trees and voted on.
"""


from __future__ import division
import numpy as np
from scipy.stats import mode
from utilities import shuffle_in_unison
from decisiontree import DecisionTreeClassifier # preddefined decision tree class



class RandomForestClassifier(object):
    

    def __init__(self, n_estimators=32, max_features=np.sqrt, max_depth=10,
        min_samples_split=2, bootstrap=0.9):
        """ Initialize tree elements for classification. """
        self.n_estimators = n_estimators # number of decision trees
        self.max_features = max_features # number of considered features
        self.max_depth = max_depth # max tree depth
        self.min_samples_split = min_samples_split # samples needed at node for node split
        self.bootstrap = bootstrap # distribution per tree
        self.forest = []


    def fit(self, P, q):
        """ 
        Data divided into randomized subsets creating forest of decision trees
        used for classification.
        """
        self.forest = []
        n_samples = len(q)
        n_sub = round(n_samples*self.bootstrap)
        
        for i in xrange(self.n_estimators):
            shuffle_in_unison(P, q)
            P_subset = P[:n_sub]
            q_subset = q[:n_sub]

            tree = DecisionTreeClassifier(self.max_features, self.max_depth, self.min_samples_split)
            tree.fit(P_subset, q_subset)
            self.forest.append(tree)


    def predict(self, A):
        """ Class prediction in trees of A """
        n_samples = A.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in xrange(n_trees):
            predictions[i] = self.forest[i].predict(A)

        return mode(predictions)[0][0]


    def score(self, P, q):
        """ Return the accuracy of the prediction of X compared to y. """
        q_pred = self.predict(P)
        n_samples = len(q)
        correct = 0
        for j in xrange(n_samples):
            if q_pred[j] == q[j]:
                correct += 1
        return correct / n_samples
