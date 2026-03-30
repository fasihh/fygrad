from node import Device, xp


class DataLoader:
    def __init__(
        self,
        X, y,
        batch_size=32,
        shuffle=True,
        device: Device = "cpu"
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __convert_to_device(self, device: Device):
        self.X = xp(device).asarray(self.X)
        self.y = xp(device).asarray(self.y)

    def to_gpu(self):
        self.__convert_to_device("gpu")
        self.device = "gpu"
        return self
    
    def to_cpu(self):
        self.__convert_to_device("cpu")
        self.device = "cpu"
        return self

    def __iter__(self):
        idx = xp(self.device).arange(len(self.X))

        if self.shuffle:
            xp(self.device).random.shuffle(idx)

        for i in range(0, len(idx), self.batch_size):
            batch = idx[i : i + self.batch_size]
            X = self.X[batch]
            y = self.y[batch]
            yield X, y
