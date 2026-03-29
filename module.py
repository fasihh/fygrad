from typing import Dict, List
from node import Node

class Module:
    def __init__(self):
        self._parameters: Dict[str, Node] = {}
        self._modules: Dict[str, "Module"] = {}
        self._cached_params: List[Node] | None = None

    def parameters(self) -> List[Node]:
        if self._cached_params is not None:
            return self._cached_params
        
        params = [p for p in self._parameters.values()]
        for m in self._modules.values():
            params.extend(m.parameters())
        self._cached_params = params
        return self._cached_params

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

    def forward(self, _):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, label: str = ""):
        super().__init__()

        self.W = Node.randn((in_dim, out_dim), 0.1)
        self.W.label = f"{label}.W" if label else "W"

        self.b = Node.randn((1, out_dim), 0.1)
        self.b.label = f"{label}.b" if label else "b"

    def forward(self, x):
        return Node.matmul(x, self.W) + self.b
