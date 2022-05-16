class Cost:
    def __init__(self) -> None:
        pass
    
    def LinearRegressionCost(self,x,y, theta):
        return ((hypothesis(x,theta) - y).T@(hypothesis(x,theta)-y)/(2*y.shape[0])) 
        
    def LogisticRegressionCost(self):
        return None
    def LassoRegressionCost(self):
        return None
    def RidgeRegressionCost(self):
        return None