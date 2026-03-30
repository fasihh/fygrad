from typing import List, Union, Literal
import numpy as np

type Value = Union["Node", float, np.ndarray]
type Device = Literal["cpu", "gpu"]

def xp(device: Device):
    if device == "cpu":
        import numpy as np
        return np
    try:
        import cupy as cp # type: ignore
        return cp
    except ImportError:
        raise RuntimeError("cuda environment missing")

class Node:
    def __init__(
        self,
        label: str, value: np.ndarray,
        grad: np.ndarray = None,
        children: List["Node"] = [],
        device: Device = "cpu"
    ):
        self.label = label
        self.value = value
        self.grad = grad
        self.children = children
        self.device = device
        self._backward = lambda: None

        if xp(device).isscalar(self.value):
            self.value = xp(device).array([[self.value]], dtype=xp(device).float64)
        else:
            self.value = xp(device).asarray(self.value, dtype=xp(device).float64)
        if self.grad is None:
            self.grad = xp(device).zeros_like(self.value, dtype=xp(device).float64)

    def __convert_to_device(self, device: Device):
        self.device = device

        self.value = xp(device).asarray(self.value)
        self.grad = xp(device).asarray(self.grad)
        self.device = "gpu"

        for child in self.children:
            child.__convert_to_device(device)

    def to_gpu(self):
        self.__convert_to_device("gpu")
        return self
    
    def to_cpu(self):
        self.__convert_to_device("cpu")
        return self

    def zero_grad(self):
        self.grad = xp(self.device).zeros_like(self.grad, dtype=xp(self.device).float64)

    @staticmethod
    def __ensure_node(obj: Value, device: Device = "cpu") -> "Node":
        if isinstance(obj, Node):
            if obj.device != device:
                raise RuntimeError(f"mismatch in devices: {obj} not {device}")
            return obj
        val = xp(device).array([[obj]], dtype=xp(device).float64) if xp(device).isscalar(obj) else xp(device).array(obj, dtype=xp(device).float64)
        return Node(str(obj), val, device=device)

    @staticmethod
    def __sum_to_shape(grad: np.ndarray, shape: tuple) -> np.ndarray:
        """Reduce a broadcasted gradient back to the original operand shape."""
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad
    
    @staticmethod
    def ones(label: str, shape: tuple, device: Device = "cpu") -> "Node":
        return Node(label, xp(device).ones(shape), device=device)
    
    @staticmethod
    def zeros(label: str, shape: tuple, device: Device = "cpu") -> "Node":
        return Node(label, xp(device).zeros(shape), device=device)
    
    @staticmethod
    def randn(label: str, shape: tuple, scale: float = 1.0, device: Device = "cpu") -> "Node":
        return Node(label, xp(device).random.randn(*shape) * scale, device=device)

    @staticmethod
    def tanh(obj: Value, device: Device = "cpu") -> "Node":
        obj = Node.__ensure_node(obj, device)
        value = 2 * (1 / (1 + xp(device).exp(2 * -obj.value))) - 1
        out = Node(f"tanh({obj.label})", value=value, children=[obj], device=device)

        def _backward():
            obj.grad += out.grad * (1 - value ** 2)
        
        out._backward = _backward
        return out
    
    @staticmethod
    def relu(obj: Value, device: Device = "cpu") -> "Node":
        obj = Node.__ensure_node(obj, device)
        value = xp(device).maximum(0, obj.value)
        out = Node(f"relu({obj.label})", value=value, children=[obj], device=device)

        def _backward():
            obj.grad += out.grad * (obj.value > 0)
        
        out._backward = _backward
        return out

    @staticmethod
    def sigmoid(obj: Value, device: Device = "cpu") -> "Node":
        obj = Node.__ensure_node(obj, device)
        value = 1 / (1 + xp(device).exp(-obj.value))
        out = Node(f"sigmoid({obj.label})", value, children=[obj], device=device)

        def _backward():
            obj.grad += out.grad * value * (1 - value)

        out._backward = _backward
        return out

    @staticmethod
    def softmax(obj: Value, device: Device = "cpu") -> "Node":
        obj = Node.__ensure_node(obj, device)
        shift = xp(device).max(obj.value, keepdims=True, axis=1)
        exps = xp(device).exp(obj.value - shift)
        probs = exps / xp(device).sum(exps, keepdims=True, axis=1)
        out = Node(f"softmax({obj.label})", probs, children=[obj], device=device)

        def _backward():
            # s = probs.reshape(-1,1)
            # jac = np.diagflat(probs) - np.dot(s, s.T)
            # obj.grad += jac @ out.grad.T
            v_dot_p = xp(device).sum(out.grad * probs, axis=1, keepdims=True)
            obj.grad += probs * (out.grad - v_dot_p)
        
        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(probs: Value, target_indices: np.ndarray, device: Device = "cpu") -> "Node":
        probs = Node.__ensure_node(probs, device)
        batch_size = probs.value.shape[0]
        correct_confidences = probs.value[xp(device).arange(batch_size), target_indices]
        loss_value = -xp(device).mean(xp(device).log(correct_confidences))

        out = Node(f"cross_entropy({probs.label})", xp(device).array([[loss_value]]), children=[probs], device=device)

        def _backward():
            grad = xp(device).zeros_like(probs.value)
            grad[xp(device).arange(batch_size), target_indices] = -1.0 / (correct_confidences + 1e-15)
            probs.grad += (grad * out.grad) / batch_size

        out._backward = _backward
        return out

    @staticmethod
    def binary_cross_entropy(prob: Value, target: np.ndarray, device: Device = "cpu") -> "Node":
        prob = Node.__ensure_node(prob, device)
        target = Node.__ensure_node(target, device)
        xpv = xp(device)
        loss_value = -xpv.mean(target.value * xpv.log(prob.value) + (1 - target.value) * xpv.log(1 - prob.value))

        out = Node(f"binary_cross_entropy({prob.label})", value=loss_value, children=[prob], device=device)

        def _backward():
            prob.grad += out.grad * (prob.value - target.value) / (prob.value * (1 - prob.value))

        out._backward = _backward
        return out
    
    @staticmethod
    def mse(values: Value, target: Value, device: Device = "cpu") -> "Node":
        values = Node.__ensure_node(values, device)
        target = Node.__ensure_node(target, device)
        batch_size = values.value.shape[0]
        loss_value = xp(device).mean((values.value[xp(device).arange(batch_size)] - target.value) ** 2) / 2

        out = Node(f"mse({values.label})", xp(device).array([[loss_value]]), children=[values], device=device)

        def _backward():
            grad = values.value[xp(device).arange(batch_size)] - target.value
            values.grad += out.grad * grad

        out._backward = _backward
        return out

    @staticmethod
    def matmul(a: Value, b: Value, device: Device = "cpu") -> "Node":
        a = Node.__ensure_node(a, device)
        b = Node.__ensure_node(b, device)
        value = a.value @ b.value
        out = Node(f"matmul({a.label}, {b.label})", value, children=[a,b], device=device)

        def _backward():
            a.grad += out.grad @ b.value.T
            b.grad += a.value.T @ out.grad

        out._backward = _backward
        return out
    
    @staticmethod
    def concat(a: Value, b: Value, axis: int = 1, device: Device = "cpu") -> "Node":
        a, b = Node.__ensure_node(a, device), Node.__ensure_node(b, device)
        value = xp(device).concatenate((a.value, b.value), axis=axis)
        out = Node(f"concat({a.label}, {b.label})", value, children=[a, b], device=device)

        def _backward():
            split = a.value.shape[axis]

            if axis == 1:
                a.grad += out.grad[:, :split]
                b.grad += out.grad[:, split:]
            else:
                a.grad += out.grad[:split]
                b.grad += out.grad[split:]
            
        out._backward = _backward
        return out

    def sum(self) -> "Node":
        value = xp(self.device).sum(self.value)
        out = Node(f"sum({self.label})", value=value, children=[self], device=self.device)

        def _backward():
            self.grad += out.grad * xp(self.device).ones_like(self.value)
        
        out._backward = _backward
        return out
    
    def abs(self) -> "Node":
        value = xp(self.device).abs(self.value)
        out = Node(f"abs({self.label})", value=value, children=[self], device=self.device)

        def _backward():
            self.grad += out.grad * xp(self.device).sign(self.value)

        out._backward = _backward
        return out

    def __add__(self, obj: Value) -> "Node":
        obj = Node.__ensure_node(obj, self.device)
        out = Node(f"({self.label}+{obj.label})", self.value + obj.value, children=[self,obj], device=self.device)

        def _backward():
            self.grad += Node.__sum_to_shape(out.grad, self.value.shape)
            obj.grad += Node.__sum_to_shape(out.grad, obj.value.shape)

        out._backward = _backward
        return out

    def __mul__(self, obj: Value) -> "Node":
        obj = Node.__ensure_node(obj, self.device)
        out = Node(f"({self.label}*{obj.label})", self.value * obj.value, children=[self,obj], device=self.device)

        def _backward():
            self.grad += Node.__sum_to_shape(out.grad * obj.value, self.value.shape)
            obj.grad += Node.__sum_to_shape(out.grad * self.value, obj.value.shape)

        out._backward = _backward
        return out

    def __neg__(self) -> "Node":
        return self * Node("-1", -1, device=self.device)

    def __sub__(self, obj: Value) -> "Node":
        return self + (-Node.__ensure_node(obj, self.device))
    
    def __truediv__(self, obj: Value) -> "Node":
        return self * (Node.__ensure_node(obj, self.device) ** -1)
    
    def __pow__(self, power: float) -> "Node":
        out = Node(f"({self.label}^{power})", self.value ** power, children=[self], device=self.device)

        def _backward():
            self.grad += out.grad * power * (self.value ** (power - 1))
        
        out._backward = _backward
        return out
    
    def __str__(self):
        return f"{self.label}={self.value}"

    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return id(self)

    def __len__(self):
        return len(self.value)

    @property
    def shape(self):
        return self.value.shape

    def backward(self):
        topo = []
        visited = set()
        def build(node):
            if node in visited:
                return
            visited.add(node)
            for child in node.children:
                build(child)
            topo.append(node)
        build(self)
        self.grad = xp(self.device).ones_like(self.value, dtype=xp(self.device).float64)
        for node in reversed(topo):
            node._backward()
