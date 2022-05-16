import numpy as np
from Internals.GradientDescent import GradientDescent
from Internals.Cost import Cost

class LinearRegression:
    def __init__(self) -> None:
        self.beta = None
        self.mu = []
        self.std = []
        pass
    
    def normalize(self, data):
        for i in range(1,data.shape[1]):
            data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))
            self.mu.append(np.mean(data[:,i]))
            self.std.append(np.std(data[:, i]))     

    def fit(self, X,y):

        self.X = X
        self.Y = y
        theta = np.zeros((self.X.shape[1],1))

        def hypothesis(X, theta):
                return np.matmul(X,theta)
            
        def cost(x,y, theta):
            return ((hypothesis(x,theta) - y).T@(hypothesis(x,theta)-y)/(2*y.shape[0])) 
        
        def gradient_descent(x,y, theta, learning_rate = 0.1, num_epocs = 10):
            m = x.shape[0]
            J_all = []

            for _ in range(num_epocs):
                h_x = hypothesis(x, theta)
                cost_ = (1/m)*(x.T@(h_x - y))
                theta = theta - (learning_rate)*cost_
                J_all.append(cost(x, y, theta))
            
            return theta, J_all
        
        theta, J_all = gradient_descent(self.X, self.y, self.theta,learning_rate = 0.01, num_epocs = 500)
        self.beta = theta

    def predict(self, X,y):
        return None
    
        
    
            
            
        
