class GradientDescent:
    def __init__(self) -> None:
        pass

    def gradient_descent(self, x,y, theta, h_x, cost, cost_function ,learning_rate = 0.1, num_epocs = 50):
        m = x.shape[0]
        J_all = []

        for _ in range(num_epocs):
            h_x = self.c.Hypothesis(x, theta)
            cost_ = (1/m)*(x.T@(h_x - y))
            theta = theta - (learning_rate)*cost
            J_all.append(cost_function)
        
        return theta, J_all
    