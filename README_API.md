# fygrad API documentation

This document covers all public functions in `fygrad.functional` and all modules in `fygrad.module`.

## fygrad.functional

All functions accept `device="cpu"|"gpu"` and return a `Node`.

### Basic ops

- `add(a, b, device="cpu")`
  - Adds two nodes or values with broadcasting.
- `sub(a, b, device="cpu")`
  - Subtracts `b` from `a`.
- `mul(a, b, device="cpu")`
  - Elementwise multiply.
- `div(a, b, device="cpu")`
  - Elementwise divide.
- `neg(a, device="cpu")`
  - Unary negation.
- `pow(a, b, device="cpu")`
  - Elementwise power.
- `matmul(a, b, device="cpu")`
  - Matrix multiply (`@`).
- `concat(a, b, axis=1, device="cpu")`
  - Concatenate along an axis.

### Elementwise and reductions

- `exp(a, device="cpu")`
  - Elementwise exponential.
- `log(a, device="cpu")`
  - Elementwise natural log.
- `sqrt(a, device="cpu")`
  - Elementwise square root.
- `tanh(a, device="cpu")`
  - Hyperbolic tangent.
- `relu(a, device="cpu")`
  - ReLU activation.
- `sigmoid(a, device="cpu")`
  - Sigmoid activation.
- `softmax(a, axis=-1, device="cpu")`
  - Softmax over the given axis.
- `sum(a, axis=None, keepdims=False, device="cpu")`
  - Sum over an axis.
- `mean(a, axis=-1, keepdims=True, device="cpu")`
  - Mean over an axis.
- `abs(a, device="cpu")`
  - Elementwise absolute value.

### Shape helpers

- `transpose(a, device="cpu")`
  - Matrix transpose.
- `getitem(a, idx, device="cpu")`
  - Slice/index selection.
- `flatten(a, device="cpu")`
  - Flatten to shape `(batch, -1)`.

### Embedding and convolution

- `embedding(indices, weight, device="cpu")`
  - Look up vectors from a weight matrix using integer indices.
- `conv(x, kernel, stride=1, padding=0, device="cpu")`
  - 2D convolution for NCHW tensors.
- `max_pool2d(x, kernel_size, stride=None, padding=0, device="cpu")`
  - Max pooling.
- `avg_pool2d(x, kernel_size, stride=None, padding=0, device="cpu")`
  - Average pooling.

### Losses

- `mse(values, target, device="cpu")`
  - Mean squared error (scaled by 1/2).
- `cross_entropy(probs, target_indices, device="cpu")`
  - Cross entropy for probabilities and target class indices.
- `binary_cross_entropy(prob, target, device="cpu")`
  - Binary cross entropy for probabilities in `[0, 1]`.

## fygrad.module

All modules inherit `Module` and implement `forward()`.

### Base

- `Module(device="cpu")`
  - Base class. Tracks parameters and submodules.
  - `parameters()` returns a flat list of `Node` parameters.
  - `modules()` returns child modules.
  - `to_cpu()` and `to_gpu()` move parameters.
  - `state_dict()`, `load_state_dict()`, `save()`, `load()` for persistence.

### Layers

- `Linear(in_dim, out_dim, label="", device="cpu")`
  - Fully connected layer, computes `x @ W + b`.

- `RNN(input_size, hidden_size, device="cpu")`
  - Basic tanh RNN. Expects a list of time-step inputs.

- `LSTM(input_size, hidden_size, device="cpu")`
  - LSTM over a list of time-step inputs.

- `Embedding(shape, label="", device="cpu")`
  - Embedding table. `shape` is `(vocab, dim)`.

- `PositionalEncoding(d_model, max_seq_len=512, device="cpu")`
  - Sinusoidal positional encodings added to inputs.

- `LayerNorm(d_model, eps=1e-6, device="cpu")`
  - Layer normalization over the last dimension.

- `ScaledDotProductAttention()`
  - Computes attention weights and output. Optional mask is additive.

- `Conv(in_channels, out_channels, kernel_size, stride=1, padding=0, device="cpu")`
  - 2D convolution for NCHW tensors.

- `MaxPool2d(kernel_size, stride=None, padding=0, device="cpu")`
  - Max pooling.

- `AvgPool2d(kernel_size, stride=None, padding=0, device="cpu")`
  - Average pooling.

### Activations

- `Sigmoid()`
  - Sigmoid activation.
- `Tanh()`
  - Tanh activation.
- `ReLU()`
  - ReLU activation.
- `Softmax()`
  - Softmax activation over the last axis.
