from fygrad.data import Data
from fygrad import functional as F
from typing import Callable, List, Literal


class Node:
    __slots__ = (
        "label",
        "value",
        "grad",
        "children",
        "device",
        "_backward",
        "requires_grad",
    )

    def __init__(
        self,
        label: str,
        value,
        children: List["Node"] = None,
        device: Literal["cpu", "gpu"] = "cpu",
        requires_grad: bool = True,
    ):
        self.label = label
        self.value = Data(value, device=device)
        self.grad = None
        self.children: List[Node] = children or []
        self.device = device
        self._backward: Callable = lambda: None
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.value.shape

    def __len__(self):
        return len(self.value)

    @staticmethod
    def _add_grad(node: "Node", delta: Data):
        if not node.requires_grad:
            return
        if node.grad is None:
            node.grad = Data.zeros_like(node.value.shape, device=node.device)
        node.grad += delta

    def __str__(self):
        return f"{self.label}={self.value}"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return F.add(self, other, device=self.device)

    def __radd__(self, other):
        return F.add(other, self, device=self.device)

    def __sub__(self, other):
        return F.sub(self, other, device=self.device)

    def __rsub__(self, other):
        return F.sub(other, self, device=self.device)

    def __mul__(self, other):
        return F.mul(self, other, device=self.device)

    def __rmul__(self, other):
        return F.mul(other, self, device=self.device)

    def __truediv__(self, other):
        return F.div(self, other, device=self.device)

    def __rtruediv__(self, other):
        return F.div(other, self, device=self.device)

    def __pow__(self, other):
        return F.pow(self, other, device=self.device)

    def __rpow__(self, other):
        return F.pow(other, self, device=self.device)

    def __matmul__(self, other):
        return F.matmul(self, other, device=self.device)

    def __rmatmul__(self, other):
        return F.matmul(other, self, device=self.device)

    def __neg__(self):
        return F.neg(self, device=self.device)

    def exp(self):
        return F.exp(self, device=self.device)

    def tanh(self):
        return F.tanh(self, device=self.device)

    def relu(self):
        return F.relu(self, device=self.device)

    def sigmoid(self):
        return F.sigmoid(self, device=self.device)

    def log(self):
        return F.log(self, device=self.device)

    def sqrt(self):
        return F.sqrt(self, device=self.device)

    def softmax(self, axis: int = -1):
        return F.softmax(self, axis=axis, device=self.device)

    def sum(self, axis=None, keepdims: bool = False):
        return F.sum(self, axis=axis, keepdims=keepdims, device=self.device)

    def mean(self, axis: int = -1, keepdims: bool = True):
        return F.mean(self, axis=axis, keepdims=keepdims, device=self.device)

    def max_pool2d(self, kernel_size, stride=None, padding: int = 0):
        return F.max_pool2d(
            self, kernel_size, stride=stride, padding=padding, device=self.device
        )

    def avg_pool2d(self, kernel_size, stride=None, padding: int = 0):
        return F.avg_pool2d(
            self, kernel_size, stride=stride, padding=padding, device=self.device
        )

    def abs(self):
        return F.abs(self, device=self.device)

    def __getitem__(self, idx):
        return F.getitem(self, idx, device=self.device)

    @property
    def T(self):
        return F.transpose(self, device=self.device)

    def zero_grad(self):
        if self.grad is not None:
            self.grad = Data.zeros_like(self.value.shape, device=self.device)

    def to_gpu(self):
        self.device = "gpu"
        self.value = Data(self.value.data, device="gpu")
        if self.grad is not None:
            self.grad = Data(self.grad.data, device="gpu")
        for child in self.children:
            child.to_gpu()
        return self

    def to_cpu(self):
        self.device = "cpu"
        self.value = Data(self.value.data, device="cpu")
        if self.grad is not None:
            self.grad = Data(self.grad.data, device="cpu")
        for child in self.children:
            child.to_cpu()
        return self

    def state_dict(self):
        return {
            "value": self.value.data.tolist(),
            "shape": self.value.shape,
            "device": self.device,
        }

    def load_state_dict(self, state):
        self.device = state["device"]
        self.value = Data(state["value"], device=self.device)

    def backward(self):
        if not self.requires_grad:
            return
        topo: List[Node] = []
        visited = set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            for child in node.children:
                build(child)
            topo.append(node)

        build(self)

        self.grad = Data.ones_like(self.value.shape, device=self.device)
        for node in reversed(topo):
            if not node.requires_grad or node.grad is None:
                continue
            node._backward()

    @staticmethod
    def ones(label: str, shape: tuple, device: Literal["cpu", "gpu"] = "cpu"):
        return Node(label, Data.ones_like(shape, device=device), device=device)

    @staticmethod
    def zeros(label: str, shape: tuple, device: Literal["cpu", "gpu"] = "cpu"):
        return Node(label, Data.zeros_like(shape, device=device), device=device)

    @staticmethod
    def randn(
        label: str,
        shape: tuple,
        scale: float = 1.0,
        device: Literal["cpu", "gpu"] = "cpu",
    ):
        return Node(
            label,
            Data.randn(shape, device=device) * Data(scale, device=device),
            device=device,
        )
