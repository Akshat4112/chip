import numpy as np

class Cost:
    def __init__(self) -> None:
        pass
    
    def Hypothesis(self, X, theta):
        self.X = X
        self.theta = theta
        return np.matmul(self.X, self.theta)

    def LinearRegressionCost(self,x,y, theta):
        return ((self.Hypothesis(x,theta) - y).T@(self.Hypothesis(x,theta)-y)/(2*y.shape[0])) 
        
    def LogisticRegressionCost(self):
        return None
    def LassoRegressionCost(self):
        return None
    def RidgeRegressionCost(self):
        return None