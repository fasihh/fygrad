import json
from typing import Dict, List, Any
from fygrad.node import Node, Device, xp


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

    def state_dict(self):
        state = {}

        for name, param in self._parameters.items():
            state[name] = param.state_dict()

        for name, module in self._modules.items():
            state[name] = module.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self._cached_params = None

        for name, param in self._parameters.items():
            param.load_state_dict(state[name])

        for name, module in self._modules.items():
            module.load_state_dict(state[name])

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.state_dict(), f)

    def load(self, filename: str):
        with open(filename, "r") as f:
            self.load_state_dict(json.load(f))

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

    def forward(self, *_, **__) -> Node:
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
        device: Device = "cpu",
    ):
        super().__init__(device)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = Node.randn("Wx", (input_size, hidden_size), device=device)
        self.Wh = Node.randn("Wh", (hidden_size, hidden_size), device=device)
        self.bh = Node.zeros("bh", (1, hidden_size), device=device)

    def forward(self, seq):
        if not isinstance(seq, list):
            raise RuntimeError("RNN.forward takes a sequence of inputs")

        batch_size = seq[0].shape[0]
        hidden = Node.zeros("h", (batch_size, self.hidden_size), device=self.device)
        hidden_states: List[Node] = []

        for x in seq:
            hidden = Node.tanh(
                Node.matmul(x, self.Wx, self.device)
                + Node.matmul(hidden, self.Wh, self.device)
                + self.bh,
                self.device,
            )
            hidden_states.append(hidden)

        return hidden_states


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, device: Device = "cpu"):
        super().__init__(device)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wi = Node.randn("Wi", (input_size, hidden_size), device=device)
        self.Wf = Node.randn("Wf", (input_size, hidden_size), device=device)
        self.Wo = Node.randn("Wo", (input_size, hidden_size), device=device)
        self.Wg = Node.randn("Wg", (input_size, hidden_size), device=device)

        self.Ui = Node.randn("Ui", (hidden_size, hidden_size), device=device)
        self.Uf = Node.randn("Uf", (hidden_size, hidden_size), device=device)
        self.Uo = Node.randn("Uo", (hidden_size, hidden_size), device=device)
        self.Ug = Node.randn("Ug", (hidden_size, hidden_size), device=device)

        self.bi = Node.zeros("bi", (1, hidden_size), device=device)
        self.bf = Node.zeros("bf", (1, hidden_size), device=device)
        self.bo = Node.zeros("bo", (1, hidden_size), device=device)
        self.bg = Node.zeros("bg", (1, hidden_size), device=device)

    def forward(self, seq):
        if not isinstance(seq, list):
            raise RuntimeError("LSTM.forward takes a sequence of inputs")

        batch_size = seq[0].shape[0]
        h = Node.zeros("h", (batch_size, self.hidden_size), device=self.device)
        cstate = Node.zeros(
            "cstate", (batch_size, self.hidden_size), device=self.device
        )
        hidden_states: List[Node] = []

        for x in seq:
            i = Node.sigmoid(
                Node.matmul(x, self.Wi, self.device)
                + Node.matmul(h, self.Ui, self.device)
                + self.bi,
                self.device,
            )
            f = Node.sigmoid(
                Node.matmul(x, self.Wf, self.device)
                + Node.matmul(h, self.Uf, self.device)
                + self.bf,
                self.device,
            )
            o = Node.sigmoid(
                Node.matmul(x, self.Wo, self.device)
                + Node.matmul(h, self.Uo, self.device)
                + self.bo,
                self.device,
            )

            ct = Node.tanh(
                Node.matmul(x, self.Wg, self.device)
                + Node.matmul(h, self.Ug, self.device)
                + self.bg,
                self.device,
            )

            cstate = f * cstate + i * ct
            h = Node.tanh(cstate, self.device) * o

            hidden_states.append(h)

        return hidden_states


class Embedding(Module):
    def __init__(self, shape: tuple, label: str = "", device: Device = "cpu"):
        super().__init__(device)
        self.emb = Node.randn(
            f"{label}.embed", shape, scale=xp(device).sqrt(1 / shape[-1])
        )

    def forward(self, x):
        return Node.embedding(x, self.emb, device=self.device) * xp(self.device).sqrt(
            self.emb.shape[-1]
        )


class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, device: Device = "cpu"):
        super().__init__(device)

        self.d_model = d_model
        self.pe = xp(device).zeros((max_seq_len, d_model))
        pos = xp(device).arange(max_seq_len).reshape(-1, 1)

        div = xp(device).exp(
            xp(device).arange(0, d_model, 2) * -(xp(device).log(10000.0) / d_model)
        )

        self.pe[:, 0::2] = xp(device).sin(pos * div)
        self.pe[:, 1::2] = xp(device).cos(pos * div)

    def forward(self, x):
        pe = Node("pe", self.pe[: x.shape[0]], device=self.device)
        return pe + x


class LayerNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-6, device: Device = "cpu"):
        super().__init__(device)

        self.gamma = Node.ones("gamma", (1, d_model), device=device)
        self.beta = Node.zeros("beta", (1, d_model), device=device)
        self.eps = eps

    def forward(self, x: Node):
        mean = Node.mean(x, axis=-1, device=self.device)
        diff = x - mean
        var = Node.mean(diff**2, axis=-1, device=self.device)
        x_norm = diff * ((var + self.eps) ** -0.5)
        return self.gamma * x_norm + self.beta


class ScaledDotProductAttention(Module):
    def forward(self, Q: Node, K: Node, V: Node, mask=None):
        d_k = Q.shape[-1]
        scale = 1 / xp(self.device).sqrt(d_k)
        scores = Node.matmul(Q, K.T, device=self.device) * scale
        if mask is not None:
            scores = scores + Node("mask", mask, device=self.device)

        weights = Node.softmax(scores, device=self.device)
        return Node.matmul(weights, V, device=self.device)


class Conv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        device: Device = "cpu",
    ):
        super().__init__(device)
        self.stride = stride
        self.padding = padding

        scale = xp(device).sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = Node.randn(
            "W",
            (out_channels, in_channels, kernel_size, kernel_size),
            scale=scale,
            device=device,
        )
        self.b = Node.zeros("b", (1, out_channels, 1, 1), device=device)

    def forward(self, x):
        out = Node.conv(x, self.W, self.stride, self.padding, self.device)
        return out + self.b


# Activations


class Sigmoid(Module):
    def forward(self, x):
        return Node.sigmoid(x, device=self.device)


class Tanh(Module):
    def forward(self, x):
        return Node.tanh(x, device=self.device)


class ReLU(Module):
    def forward(self, x):
        return Node.relu(x, device=self.device)


class Softmax(Module):
    def forward(self, x):
        return Node.softmax(x, device=self.device)
