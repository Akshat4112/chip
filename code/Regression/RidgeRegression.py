import sys

sys.path.append('../')

import numpy as np
from Internals.GradientDescent import GradientDescent
from Internals.Cost import Cost

class LinearRegression():
    def __init__(self) -> None:
        self.beta = None
        self.mu = []
        self.std = []
        self.c = Cost()
        self.g = GradientDescent()
        pass
    
    def fit(self, X,Y):
        
        self.X = X
        self.Y = Y
        
        self.mu.append(np.mean(self.X, axis=1))
        self.std.append(np.std(self.X, axis=1))
        self.X = self.X.T
        self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
        self.Y = np.reshape(self.Y, (10,1))
        theta = np.zeros((self.X.shape[1],1))
        
        def gradient_descent(x,y, theta, learning_rate = 0.1, num_epocs = 500):
            m = x.shape[0]
            J_all = []

            for _ in range(num_epocs):
                h_x = self.c.Hypothesis(x, theta)
                cost_ = (1/m)*(x.T@(h_x - y))
                theta = theta - (learning_rate)*cost_
                J_all.append(self.c.RidgeRegressionCost(x,y, theta, 10))
            
            return theta, J_all
        
        theta, J_all = gradient_descent(self.X, self.Y, theta, learning_rate = 0.01, num_epocs = 50)
        self.beta = theta
        print("Theta Shape After GD", theta.shape)
        print(self.beta)
        

    def predict(self, X):
        self.X = X
        # self.X[0] = (self.X[0] - self.mu[0])/self.std[0]
        y_pred = self.beta[0] + self.beta[1]*self.X[0] 
        # print(self.beta)
        print("Prediction is: ", y_pred)
        return y_pred
    
        
X = np.array([[1,2,3,4,5,6,7,8,9,10]])
Y = np.array([[10,20,30,40,50,60,70,80,90,100]])

obj = LinearRegression()
obj.fit(X, Y)
obj.predict([[50]])

            
            
        
