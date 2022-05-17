import numpy as np

class Normalize:
    def __init__(self) -> None:
        pass

    def normalize(self, data):
        self.mu = mu
        self.std = std
        for i in range(1,data.shape[1]):
            data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))
            self.mu.append(np.mean(data[:,i]))
            self.std.append(np.std(data[:, i]))     