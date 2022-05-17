import numpy as np

from code.Internals import Hypothesis

class Cost:
    def __init__(self) -> None:
        pass
    
    def Hypothesis(self, X, theta):
        self.X = X
        self.theta = theta
        return np.matmul(self.X, self.theta)

    def LinearRegressionCost(self,x,y, theta):
        return ((self.Hypothesis(x,theta) - y).T@(self.Hypothesis(x,theta)-y)/(2*y.shape[0])) 
        
    def LogisticRegressionCost(self,x,y, theta):
        return ((-y*np.log(Hypothesis(x, theta))) - (1-y)*(np.log(1- Hypothesis(x,theta))))

    def LassoRegressionCost(self,x,y, theta, lambda_=0):
        return ((self.Hypothesis(x,theta) - y).T@(self.Hypothesis(x,theta)-y)/(2*y.shape[0])) + lambda_*np.sum(theta)
        
    def RidgeRegressionCost(self,x,y, theta, lambda_=0):
        return ((self.Hypothesis(x,theta) - y).T@(self.Hypothesis(x,theta)-y)/(2*y.shape[0])) + lambda_*np.sum(theta*theta)
        