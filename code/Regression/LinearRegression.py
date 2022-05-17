import sys

sys.path.append('../')

import numpy as np
from Internals.GradientDescent import GradientDescent
from Internals.Cost import Cost

class LinearRegression:
    def __init__(self) -> None:
        self.beta = None
        self.mu = []
        self.std = []
        pass
    
    def fit(self, X,Y):
        


        self.X = X
        self.Y = Y
        
        self.mu.append(np.mean(self.X, axis=1))
        self.std.append(np.std(self.X, axis=1))
        print("Mean is: ", self.mu)
        print("std is: ", self.std)
        self.X = self.X.T

        self.Y = np.reshape(self.Y, (4,1))
        self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
        
        print("Input X Shape: ", self.X.shape)
        print("Input Y Shape: ", self.Y.shape)
        
        theta = np.zeros((self.X.shape[1],1))
        print("Theta Shape: ",theta.shape)

        def hypothesis(X, theta):
                return np.matmul(X,theta)
            
        def cost(x,y, theta):
            return ((hypothesis(x,theta) - y).T@(hypothesis(x,theta)-y))/(2*y.shape[0]) 
        
        def gradient_descent(x,y, theta, learning_rate = 0.1, num_epocs = 10):
            m = x.shape[0]
            J_all = []

            for _ in range(num_epocs):
                h_x = hypothesis(x, theta)
                cost_ = (1/m)*(x.T@(h_x - y))
                theta = theta - (learning_rate)*cost_
                J_all.append(cost(x, y, theta))
            
            return theta, J_all
        
        theta, J_all = gradient_descent(self.X, self.Y, theta, learning_rate = 0.1, num_epocs = 50)
        self.beta = theta
        print("Theta Shape After GD", theta.shape)
        print(self.beta)
        

    def predict(self, X):

        self.X = X
        self.X[0] = (self.X[0] - self.mu[0])/self.std[0]
        y_pred = self.beta[0] + self.beta[1]*self.X[0] 
        # print(self.beta)
        print("Prediction is: ", y_pred)
        return y_pred
    
        
X = np.array([[1,2,3,4]])
Y = np.array([[10,20,30,40]])

obj = LinearRegression()
obj.fit(X, Y)
obj.predict([[3]])

            
            
        
