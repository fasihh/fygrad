# fygrad

fygrad is a tiny autograd and neural-network toolkit built on NumPy, with optional GPU support through CuPy. It is intentionally small so you can read the code and understand how backprop works.

## Install

```bash
pip install fygrad
```

To use the GPU, install CuPy that matches your CUDA version (example):

```bash
pip install cupy-cuda12x
```

## What is inside

- `Data`: a thin wrapper over NumPy/CuPy arrays that keeps device info.
- `Node`: a value in the computation graph with a gradient and a backward function.
- `functional`: pure functions for ops (add, matmul, relu, conv, loss, ...).
- `module`: layers and building blocks (`Linear`, `RNN`, `LSTM`, `Conv`, ...).
- `optim`: optimizers (`SGD`, `Adam`).
- `data`: simple dataset and dataloader helpers.

## Quick start (autograd)

```python
from fygrad import Node, functional as F

x = Node("x", [[1.0, 2.0], [3.0, 4.0]])
y = F.sum(x * 2)
y.backward()
print(x.grad)
```

## Core concepts

### Data

- Holds the raw array in `data` and a `device` string.
- Automatically reshapes scalars and 1D arrays into 2D.

```python
from fygrad.data import Data

a = Data([1, 2, 3])
print(a.shape)
```

### Node

- Wraps a `Data` value and stores gradients in `grad`.
- Supports operators like `+`, `-`, `*`, `/`, `@`, `**`.
- Call `backward()` on a final scalar to compute gradients.

```python
from fygrad import Node

x = Node("x", [[1.0, 2.0]])
w = Node("w", [[3.0], [4.0]])
y = x @ w
loss = y.sum()
loss.backward()
print(w.grad)
```

### functional

This module provides stateless functions. Use them when you want explicit ops.

Common ops:

- `add`, `sub`, `mul`, `div`, `pow`, `matmul`
- `exp`, `log`, `sqrt`, `tanh`, `relu`, `sigmoid`, `softmax`
- `sum`, `mean`, `abs`, `transpose`, `getitem`, `flatten`
- `embedding`, `conv`, `max_pool2d`, `avg_pool2d`
- losses: `mse`, `cross_entropy`, `binary_cross_entropy`

```python
from fygrad import Node, functional as F

x = Node("x", [[-1.0, 2.0, 0.5]])
y = F.relu(x)
```

### module

`Module` is the base class for layers. It tracks parameters and submodules.

Built-in layers:

- `Linear`, `RNN`, `LSTM`
- `Embedding`, `PositionalEncoding`, `LayerNorm`
- `ScaledDotProductAttention`
- `Conv`, `MaxPool2d`, `AvgPool2d`
- activations: `Sigmoid`, `Tanh`, `ReLU`, `Softmax`

```python
from fygrad.module import Linear
from fygrad import Node

layer = Linear(2, 1)
x = Node("x", [[1.0, 2.0]])
y = layer(x)
```

### optim

Two optimizers are included: `SGD` and `Adam`.

```python
from fygrad.module import Linear
from fygrad.optim import SGD
from fygrad import Node

model = Linear(2, 1)
opt = SGD(model.parameters(), lr=0.1)

x = Node("x", [[1.0, 2.0]])
target = Node("t", [[1.0]])
pred = model(x)
loss = (pred - target).sum()
loss.backward()
opt.step()
opt.zero_grad()
```

### data

`ArrayDataset` and `DataLoader` are minimal helpers to batch data.

```python
from fygrad.data import ArrayDataset, DataLoader

xs = [[1.0], [2.0], [3.0], [4.0]]
ys = [[2.0], [4.0], [6.0], [8.0]]

dataset = ArrayDataset(xs, ys)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for xb, yb in loader:
	print(xb, yb)
```

## A tiny training loop

```python
from fygrad import Node, functional as F
from fygrad.module import Linear
from fygrad.optim import SGD
from fygrad.data import ArrayDataset, DataLoader

dataset = ArrayDataset([[1.0], [2.0], [3.0], [4.0]], [[2.0], [4.0], [6.0], [8.0]])
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = Linear(1, 1)
opt = SGD(model.parameters(), lr=0.1)

for _ in range(100):
	for xb, yb in loader:
		x = Node("x", xb)
		y = Node("y", yb)
		pred = model(x)
		loss = F.mse(pred, y)
		loss.backward()
		opt.step()
		opt.zero_grad()
```

## GPU usage

Use `device="gpu"` when constructing `Node` or when calling module methods, then move to GPU with `to_gpu()`.

```python
from fygrad import Node

x = Node("x", [[1.0, 2.0]], device="gpu")
print(x.device)
```

If CuPy is not available, `device="gpu"` raises a runtime error.

## Saving and loading

`Module.save()` writes a JSON state, and `load()` restores it.

```python
from fygrad.module import Linear

model = Linear(2, 1)
model.save("model.json")

model2 = Linear(2, 1)
model2.load("model.json")
```

## License

MIT
