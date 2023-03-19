# Author: ChatGPT

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union

class GeneralizedAdditiveModel(BaseEstimator, RegressorMixin):
    '''This class uses a combination of decision trees and linear regression models to fit a generalized additive model. '''

    def __init__(self, n_estimators=100, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.linear = LinearRegression()
        
    def fit(self, X, y):
        '''The fit() method trains the model by iterating over a specified number of decision trees and fitting them to the residuals between the target variable and the predictions of the previously fit trees. The decision trees are then combined using a pipeline and linear regression model to fit the remaining non-linear relationships in the data.'''
        self.trees = []
        residuals = y.copy()
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions = tree.predict(X)
            residuals -= predictions
        linear_features = make_union(*self.trees)
        linear_model = make_pipeline(linear_features, self.linear)
        linear_model.fit(X, y)
        return self
    
    def predict(self, X):
        '''The predict() method uses the fitted model to make predictions on new data.'''
        tree_predictions = [tree.predict(X) for tree in self.trees]
        linear_predictions = self.linear.predict(X)
        return sum(tree_predictions) + linear_predictions