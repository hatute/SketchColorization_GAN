import numpy as np


class DataParser:
    def __init__(self, inputs, batch_size):
        self.inputs = inputs
        self.iterator = 0
        self.batch_size = batch_size
        self.m = inputs.shape[0]
        self.iteration = self.m // self.batch_size
        self.indices = np.random.permutation(self.m)

    def get_batch(self):
        if self.iterator + 1 < self.iteration:
            batch_indices = self.indices[self.iterator * self.batch_size:(self.iterator + 1) * self.batch_size]
            self.iterator += 1
        else:
            batch_indices = self.indices[self.iterator * self.batch_size:]
            self.iterator = 0
            self.indices = np.random.permutation(self.m)
        return self.inputs[batch_indices]


if __name__ == "__main__":
    pass
