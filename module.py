import json
from typing import Dict, List
from node import Node, Device, xp


class Module:
    def __init__(self, device: Device = "cpu"):
        self._parameters: Dict[str, Node] = {}
        self._modules: Dict[str, "Module"] = {}
        self._cached_params: List[Node] | None = None
        self.device = device

    def parameters(self) -> List[Node]:
        if self._cached_params is not None:
            return self._cached_params

        params = [p for p in self._parameters.values()]
        for m in self._modules.values():
            params.extend(m.parameters())
        self._cached_params = params
        return self._cached_params

    def to_gpu(self):
        self._cached_params = None
        for param in self._parameters.values():
            param.to_gpu()
        for module in self._modules.values():
            module.to_gpu()
        self.device = "gpu"
        return self

    def to_cpu(self):
        self._cached_params = None
        for param in self._parameters.values():
            param.to_cpu()
        for module in self._modules.values():
            module.to_cpu()
        self.device = "cpu"
        return self

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

    def __str__(self):
        return json.dumps(
            {"modules": str(self._modules), "parameters": str(self._parameters)},
            indent=2,
        )

    def forward(self, *_, **__):
        raise NotImplementedError


# Sub-modules


class Linear(Module):
    def __init__(
        self, in_dim: int, out_dim: int, label: str = "", device: Device = "cpu"
    ):
        super().__init__(device)

        self.W = Node.randn(
            f"{label}.W" if label else "W", (in_dim, out_dim), 0.1, device=self.device
        )
        self.b = Node.randn(
            f"{label}.b" if label else "b", (1, out_dim), 0.1, device=self.device
        )

    def forward(self, x):
        return Node.matmul(x, self.W, device=self.device) + self.b


class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: Device = "cpu",
    ):
        super().__init__(device)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wx = Node.randn("Wx", (input_size, hidden_size), device=device)
        self.Wh = Node.randn("Wh", (hidden_size, hidden_size), device=device)
        self.bh = Node.zeros("bh", (1, hidden_size), device=device)

        self.Wy = Node.randn("Wy", (hidden_size, output_size), device=device)
        self.by = Node.zeros("by", (1, output_size), device=device)

    def forward(self, seq):
        h = Node.zeros("h", (1, self.hidden_size), device=self.device)

        for i, x in enumerate(seq):
            x = Node(
                f"x{i}", xp(self.device).array(x).reshape(1, -1), device=self.device
            )
            h = Node.matmul(x, self.Wx) + Node.matmul(h, self.Wh) + self.bh
            h = Node.tanh(h)

        y = Node.matmul(h, self.Wy) + self.by
        return Node.sigmoid(y)


# Activations


class Sigmoid(Module):
    def __init__(self, device: Device = "cpu"):
        super().__init__(device)

    def forward(self, x):
        return Node.sigmoid(x, device=self.device)


class Tanh(Module):
    def __init__(self, device: Device = "cpu"):
        super().__init__(device)

    def forward(self, x):
        return Node.tanh(x, device=self.device)


class ReLU(Module):
    def __init__(self, device: Device = "cpu"):
        super().__init__(device)

    def forward(self, x):
        return Node.relu(x, device=self.device)


class Softmax(Module):
    def __init__(self, device: Device = "cpu"):
        super().__init__(device)

    def forward(self, x):
        return Node.softmax(x, device=self.device)
