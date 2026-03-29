from typing import Dict, List, Literal
from node import Node

class Module:
    def __init__(self):
        self._parameters: Dict[Node] = {}
        self._modules: Dict["Module"] = {}
        self.device: Literal["cpu", "gpu"] = "cpu"

    def parameters(self) -> List[Node]:
        params = [p for p in self._parameters.values()]
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def __setattr__(self, name, value):
        if isinstance(value, Node):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def add_parameter(self, name: str, param: Node):
        self._parameters[name] = param

    def add_module(self, name: str, module: "Module"):
        self._modules[name] = module

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def to_gpu(self):
        for param in self.parameters():
            param.to_gpu()
        self.device = "gpu"
        return self

    def forward(self, _):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, label: str = ""):
        super().__init__()

        self.W = Node.randn((in_dim + 1, out_dim), 0.1)
        self.W.label = f"{label}.W" if label else "W"

    def forward(self, x):
        X = Node.concat(x, Node.ones((len(x), 1), device=self.W.device))
        return Node.matmul(X, self.W)
