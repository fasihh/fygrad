from typing import Callable, Iterable, Literal


def xp(device: Literal["cpu", "gpu"] = "cpu"):
    if device == "cpu":
        import numpy as np

        return np
    try:
        import cupy as cp  # type: ignore

        return cp
    except ImportError:
        raise RuntimeError("cuda environment missing")


class Data:
    def __init__(self, data, device: Literal["cpu", "gpu"] = "cpu"):
        self.device = device
        xpv = xp(device)

        try:
            import cupy as cp  # type: ignore

            if isinstance(data, cp.ndarray):
                self.data = data if device == "gpu" else cp.asnumpy(data)
                self.device = "gpu" if device == "gpu" else "cpu"
                return
        except ImportError:
            pass

        if isinstance(data, Data):
            if data.device != self.device:
                self.data = xpv.asarray(data.data)
            else:
                self.data = data.data
        else:
            self.data = xpv.asarray(data, dtype=xpv.float64)
            if self.data.ndim == 0:
                self.data = self.data.reshape(1, 1)
            elif self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)

    @property
    def xp(self):
        return xp(self.device)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @staticmethod
    def ones_like(shape, device: Literal["cpu", "gpu"] = "cpu"):
        return Data(xp(device).ones(shape), device=device)

    @staticmethod
    def zeros_like(shape, device: Literal["cpu", "gpu"] = "cpu"):
        return Data(xp(device).zeros(shape), device=device)

    @staticmethod
    def randn(shape, device: Literal["cpu", "gpu"] = "cpu"):
        return Data(xp(device).random.randn(*shape), device=device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __add__(self, other):
        return Data(self.data + other.data, device=self.device)

    def __sub__(self, other):
        return Data(self.data - other.data, device=self.device)

    def __mul__(self, other):
        return Data(self.data * other.data, device=self.device)

    def __truediv__(self, other):
        return Data(self.data / other.data, device=self.device)

    def __neg__(self):
        return Data(-self.data, device=self.device)

    def __pow__(self, other):
        return Data(self.data**other.data, device=self.device)

    def __matmul__(self, other):
        return Data(self.xp.matmul(self.data, other.data), device=self.device)

    def __rmatmul__(self, other):
        return Data(self.xp.matmul(other.data, self.data), device=self.device)

    def __str__(self):
        if self.data.shape[-1] == 1:
            return str(self.xp.round(self.data.flatten()[0], 6))
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __eq__(self, other):
        return self.xp.array_equal(self.data, other.data)

    def __ne__(self, other):
        return not self.xp.array_equal(self.data, other.data)

    def __lt__(self, other):
        return self.xp.less(self.data, other.data)

    def __le__(self, other):
        return self.xp.less_equal(self.data, other.data)

    def __gt__(self, other):
        return self.xp.greater(self.data, other.data)

    def __ge__(self, other):
        return self.xp.greater_equal(self.data, other.data)

    def __abs__(self):
        return Data(self.xp.abs(self.data), device=self.device)

    def __pos__(self):
        return Data(self.data, device=self.device)

    def __round__(self):
        return Data(self.xp.round(self.data), device=self.device)

    def __floor__(self):
        return Data(self.xp.floor(self.data), device=self.device)

    def __ceil__(self):
        return Data(self.xp.ceil(self.data), device=self.device)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class ArrayDataset(Dataset):
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("ArrayDataset requires at least one tensor")
        length = len(tensors[0])
        for t in tensors[1:]:
            if len(t) != length:
                raise ValueError("All tensors must have the same length")
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)


def default_collate(batch, device: Literal["cpu", "gpu"] = "cpu"):
    xpv = xp(device)
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        transposed = list(zip(*batch))
        return tuple(
            default_collate(list(samples), device=device) for samples in transposed
        )
    return xpv.asarray(batch, dtype=xpv.float64)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Literal["cpu", "gpu"] = "cpu",
        collate_fn: Callable | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.collate_fn = collate_fn or default_collate

    def __iter__(self) -> Iterable:
        xpv = xp(self.device)
        indices = xpv.arange(len(self.dataset))
        if self.shuffle:
            xpv.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            batch = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(batch, device=self.device)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
