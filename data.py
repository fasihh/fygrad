from typing import Literal

from backend import ensure_device_available, np, xp_for_device

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    @property
    def xp(self):
        return xp_for_device(self.device)

    def to_device(self, device: Literal["cpu", "gpu"]):
        ensure_device_available(device)
        self.device = device
        return self

    def __iter__(self):
        idx = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(idx)

        for i in range(0, len(idx), self.batch_size):
            batch = idx[i : i + self.batch_size]
            X = self.dataset.X[batch]
            y = self.dataset.y[batch]
            yield self.xp.asarray(X), self.xp.asarray(y)
