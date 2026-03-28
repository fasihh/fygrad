from dataclasses import dataclass, field
from typing import List, Callable, Union
import numpy as np

Value = Union["Node", float, np.ndarray]

@dataclass
class Node:
    label: str
    value: np.ndarray
    grad: np.ndarray = None
    children: List["Node"] = field(default_factory=list)
    _backward: Callable = lambda: None

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)

    def __post_init__(self):
        if np.isscalar(self.value):
            self.value = np.array([[self.value]], dtype=np.float64)
        if self.grad is None:
            self.grad = np.zeros_like(self.value)

    @staticmethod
    def __ensure_node(obj: Value) -> "Node":
        if isinstance(obj, Node):
            return obj
        val = np.array([[obj]]) if np.isscalar(obj) else np.array(obj)
        return Node(str(obj), val)

    @staticmethod
    def relu(obj: Value) -> "Node":
        obj = Node.__ensure_node(obj)
        value = np.maximum(0, obj.value)
        out = Node(f"relu({obj.label})", value, children=[obj])

        def _backward():
            obj.grad += out.grad * (obj.value > 0)
        
        out._backward = _backward
        return out

    @staticmethod
    def sigmoid(obj: Value) -> "Node":
        obj = Node.__ensure_node(obj)
        value = 1 / (1 + np.exp(-obj.value))
        out = Node(f"sigmoid({obj.label})", value, children=[obj])

        def _backward():
            obj.grad += out.grad * value * (1 - value)

        out._backward = _backward
        return out

    @staticmethod
    def softmax(obj: Value) -> "Node":
        obj = Node.__ensure_node(obj)
        shift = np.max(obj.value, keepdims=True, axis=1)
        exps = np.exp(obj.value - shift)
        probs = exps / np.sum(exps, keepdims=True, axis=1)
        out = Node(f"softmax({obj.label})", probs, children=[obj])

        def _backward():
            # s = probs.reshape(-1,1)
            # jac = np.diagflat(probs) - np.dot(s, s.T)
            # obj.grad += jac @ out.grad.T
            v_dot_p = np.sum(out.grad * probs, axis=1, keepdims=True)
            obj.grad += probs * (out.grad - v_dot_p)
        
        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(probs: "Node", target_indices: np.ndarray) -> "Node":
        batch_size = probs.value.shape[0]
        correct_confidences = probs.value[np.arange(batch_size), target_indices]
        loss_value = -np.mean(np.log(correct_confidences))

        out = Node(f"cross_entropy({probs.label})", np.array([[loss_value]]), children=[probs])

        def _backward():
            grad = np.zeros_like(probs.value)
            grad[np.arange(batch_size), target_indices] = -1.0 / (correct_confidences + 1e-15)
            probs.grad += (grad * out.grad) / batch_size

        out._backward = _backward
        return out
    
    @staticmethod
    def mse(values: "Node", target_values: np.ndarray) -> "Node":
        batch_size = values.value.shape[0]
        loss_value = np.mean((values.value[np.arange(batch_size)] - target_values) ** 2) / 2

        out = Node(f"mse({values.label})", np.array([[loss_value]]), children=[values])

        def _backward():
            grad = values.value[np.arange(batch_size)] - target_values
            values.grad += out.grad * grad

        out._backward = _backward
        return out

    @staticmethod
    def matmul(a: Value, b: Value) -> "Node":
        a, b = Node.__ensure_node(a), Node.__ensure_node(b)
        value = a.value @ b.value
        out = Node(f"matmul({a.label}, {b.label})", value, children=[a,b])

        def _backward():
            a.grad += out.grad @ b.value.T
            b.grad += a.value.T @ out.grad

        out._backward = _backward
        return out
    
    @staticmethod
    def concat(a: Value, b: Value, axis: int = 1) -> "Node":
        a, b = Node.__ensure_node(a), Node.__ensure_node(b)
        value = np.concatenate((a.value, b.value), axis=axis)
        out = Node(f"concat({a.label}, {b.label})", value, children=[a, b])

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

    def __add__(self, obj: Value) -> "Node":
        obj = Node.__ensure_node(obj)
        out = Node(f"({self.label}+{obj.label})", self.value + obj.value, children=[self,obj])

        def _backward():
            print(self.label, obj.label, out.label, sep="\n")
            self.grad += out.grad
            obj.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, obj: Value) -> "Node":
        obj = Node.__ensure_node(obj)
        out = Node(f"({self.label}*{obj.label})", self.value * obj.value, children=[self,obj])

        def _backward():
            self.grad += out.grad * obj.value
            obj.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __neg__(self) -> "Node":
        return self * Node("-1", -1)

    def __sub__(self, obj: Value) -> "Node":
        return self + (-Node.__ensure_node(obj))
    
    def __truediv__(self, obj: Value) -> "Node":
        return self * (obj ** -1)
    
    def __pow__(self, power: float) -> "Node":
        out = Node(f"({self.label}^{power})", self.value ** power, children=[self])

        def _backward():
            self.grad += out.grad * power * (self.value ** (power - 1))
        
        out._backward = _backward
        return out
    
    def __str__(self):
        return f"{self.label}={self.value}"
    
    def __hash__(self):
        return id(self)

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
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            node._backward()
