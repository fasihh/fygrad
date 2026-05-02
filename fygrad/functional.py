from typing import Literal, TYPE_CHECKING
from fygrad.data import Data, xp

if TYPE_CHECKING:
    from fygrad.node import Node  # noqa: F401


def _ensure_node(obj, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    if isinstance(obj, Node):
        return obj
    return Node(str(obj), obj, device=device, requires_grad=False)


def _sum_to_shape(grad, shape, device: Literal["cpu", "gpu"] = "cpu"):
    xpv = xp(device)
    while grad.ndim > len(shape):
        grad = xpv.sum(grad, axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = xpv.sum(grad, axis=axis, keepdims=True)
    return grad


def _reshape_grad_to_axis(
    grad,
    axis,
    keepdims,
    original_ndim: int,
    device: Literal["cpu", "gpu"] = "cpu",
):
    if axis is None or keepdims:
        return grad
    if isinstance(axis, int):
        axis = (axis,)
    axes = tuple(a if a >= 0 else a + original_ndim for a in axis)
    for ax in sorted(axes):
        grad = xp(device).expand_dims(grad, axis=ax)
    return grad


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def add(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}+{b.label})", a.value + b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(
                a,
                Data(
                    _sum_to_shape(out.grad.data, a.value.shape, device), device=device
                ),
            )
        if b.requires_grad:
            Node._add_grad(
                b,
                Data(
                    _sum_to_shape(out.grad.data, b.value.shape, device), device=device
                ),
            )

    out._backward = backward
    return out


def sub(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}-{b.label})", a.value - b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(
                a,
                Data(
                    _sum_to_shape(out.grad.data, a.value.shape, device), device=device
                ),
            )
        if b.requires_grad:
            Node._add_grad(
                b,
                Data(
                    _sum_to_shape(-out.grad.data, b.value.shape, device), device=device
                ),
            )

    out._backward = backward
    return out


def mul(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}*{b.label})", a.value * b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            grad_a = out.grad.data * b.value.data
            Node._add_grad(
                a, Data(_sum_to_shape(grad_a, a.value.shape, device), device=device)
            )
        if b.requires_grad:
            grad_b = out.grad.data * a.value.data
            Node._add_grad(
                b, Data(_sum_to_shape(grad_b, b.value.shape, device), device=device)
            )

    out._backward = backward
    return out


def div(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}/{b.label})", a.value / b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            grad_a = out.grad.data / b.value.data
            Node._add_grad(
                a, Data(_sum_to_shape(grad_a, a.value.shape, device), device=device)
            )
        if b.requires_grad:
            grad_b = -out.grad.data * a.value.data / (b.value.data**2)
            Node._add_grad(
                b, Data(_sum_to_shape(grad_b, b.value.shape, device), device=device)
            )

    out._backward = backward
    return out


def neg(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    out = Node(f"(-{a.label})", -a.value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(a, Data(-out.grad.data, device=device))

    out._backward = backward
    return out


def pow(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}**{b.label})", a.value**b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            grad_a = out.grad.data * b.value.data * (a.value.data ** (b.value.data - 1))
            Node._add_grad(
                a, Data(_sum_to_shape(grad_a, a.value.shape, device), device=device)
            )
        if b.requires_grad:
            xpv = xp(device)
            grad_b = out.grad.data * out.value.data * xpv.log(a.value.data)
            Node._add_grad(
                b, Data(_sum_to_shape(grad_b, b.value.shape, device), device=device)
            )

    out._backward = backward
    return out


def matmul(a, b, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    out = Node(
        f"({a.label}@{b.label})", a.value @ b.value, children=[a, b], device=device
    )
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            grad_a = out.grad.data @ b.value.data.T
            Node._add_grad(
                a, Data(_sum_to_shape(grad_a, a.value.shape, device), device=device)
            )
        if b.requires_grad:
            grad_b = a.value.data.T @ out.grad.data
            Node._add_grad(
                b, Data(_sum_to_shape(grad_b, b.value.shape, device), device=device)
            )

    out._backward = backward
    return out


def concat(a, b, axis: int = 1, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    b = _ensure_node(b, device)
    xpv = xp(device)
    value = Data(
        xpv.concatenate((a.value.data, b.value.data), axis=axis), device=device
    )
    out = Node(f"concat({a.label}, {b.label})", value, children=[a, b], device=device)
    out.requires_grad = a.requires_grad or b.requires_grad

    def backward():
        if a.requires_grad:
            split = a.value.shape[axis]
            slicer = [slice(None)] * value.data.ndim
            slicer[axis] = slice(0, split)
            Node._add_grad(a, Data(out.grad.data[tuple(slicer)], device=device))
        if b.requires_grad:
            split = a.value.shape[axis]
            slicer = [slice(None)] * value.data.ndim
            slicer[axis] = slice(split, None)
            Node._add_grad(b, Data(out.grad.data[tuple(slicer)], device=device))

    out._backward = backward
    return out


def exp(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.exp(a.value.data), device=device)
    out = Node(f"exp({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(a, Data(out.grad.data * value.data, device=device))

    out._backward = backward
    return out


def log(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.log(a.value.data), device=device)
    out = Node(f"log({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data / a.value.data
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def sqrt(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.sqrt(a.value.data), device=device)
    out = Node(f"sqrt({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data * (0.5 / value.data)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def tanh(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.tanh(a.value.data), device=device)
    out = Node(f"tanh({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data * (1 - value.data**2)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def relu(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.maximum(0, a.value.data), device=device)
    out = Node(f"relu({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data * (a.value.data > 0)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def sigmoid(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(1 / (1 + xpv.exp(-a.value.data)), device=device)
    out = Node(f"sigmoid({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data * value.data * (1 - value.data)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def softmax(a, axis: int = -1, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    shift = xpv.max(a.value.data, axis=axis, keepdims=True)
    exps = xpv.exp(a.value.data - shift)
    probs = exps / xpv.sum(exps, axis=axis, keepdims=True)
    value = Data(probs, device=device)
    out = Node(f"softmax({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data
            dot = xpv.sum(grad * probs, axis=axis, keepdims=True)
            dx = probs * (grad - dot)
            Node._add_grad(a, Data(dx, device=device))

    out._backward = backward
    return out


def sum(a, axis=None, keepdims: bool = False, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.sum(a.value.data, axis=axis, keepdims=keepdims), device=device)
    out = Node(f"sum({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = _reshape_grad_to_axis(
                out.grad.data, axis, keepdims, a.value.data.ndim, device
            )
            grad = grad * xpv.ones_like(a.value.data)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def mean(
    a, axis: int = -1, keepdims: bool = True, device: Literal["cpu", "gpu"] = "cpu"
):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.mean(a.value.data, axis=axis, keepdims=keepdims), device=device)
    out = Node(f"mean({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            denom = a.value.data.shape[axis] if axis is not None else a.value.data.size
            grad = _reshape_grad_to_axis(
                out.grad.data, axis, keepdims, a.value.data.ndim, device
            )
            grad = grad * xpv.ones_like(a.value.data) / denom
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def abs(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    xpv = xp(device)
    value = Data(xpv.abs(a.value.data), device=device)
    out = Node(f"abs({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            grad = out.grad.data * xpv.sign(a.value.data)
            Node._add_grad(a, Data(grad, device=device))

    out._backward = backward
    return out


def transpose(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    value = Data(a.value.data.T, device=device)
    out = Node(f"{a.label}.T", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(a, Data(out.grad.data.T, device=device))

    out._backward = backward
    return out


def getitem(a, idx, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    value = Data(a.value.data[idx], device=device)
    out = Node(f"{a.label}[{idx}]", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            if a.grad is None:
                a.grad = Data.zeros_like(a.value.shape, device=device)
            a.grad.data[idx] += out.grad.data

    out._backward = backward
    return out


def embedding(indices, weight, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    weight = _ensure_node(weight, device)
    value = Data(weight.value.data[indices], device=device)
    out = Node(f"emb({indices})", value, children=[weight], device=device)
    out.requires_grad = weight.requires_grad

    def backward():
        if weight.requires_grad:
            if weight.grad is None:
                weight.grad = Data.zeros_like(weight.value.shape, device=device)
            xp(device).add.at(weight.grad.data, indices, out.grad.data)

    out._backward = backward
    return out


def conv(
    x, kernel, stride: int = 1, padding: int = 0, device: Literal["cpu", "gpu"] = "cpu"
):
    from fygrad.node import Node

    x = _ensure_node(x, device)
    kernel = _ensure_node(kernel, device)
    xpv = xp(device)

    batch, in_ch, h, w = x.value.shape
    out_ch, _, kh, kw = kernel.value.shape

    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    if padding > 0:
        x_padded = xpv.pad(
            x.value.data, ((0, 0), (0, 0), (padding, padding), (padding, padding))
        )
    else:
        x_padded = x.value.data

    cols = xpv.zeros((batch, in_ch * kh * kw, h_out * w_out))
    for i in range(kh):
        for j in range(kw):
            row = i * kw + j
            cols[:, row * in_ch : (row + 1) * in_ch, :] = x_padded[
                :,
                :,
                i : i + stride * h_out : stride,
                j : j + stride * w_out : stride,
            ].reshape(batch, in_ch, -1)

    w_col = kernel.value.data.reshape(out_ch, -1)
    out_val = (w_col @ cols).reshape(batch, out_ch, h_out, w_out)
    out = Node(
        f"conv({x.label}, {kernel.label})",
        Data(out_val, device=device),
        children=[x, kernel],
        device=device,
    )
    out.requires_grad = x.requires_grad or kernel.requires_grad

    def backward():
        grad_out = out.grad.data.reshape(batch, out_ch, -1)
        d_w = grad_out @ cols.transpose(0, 2, 1)
        if kernel.requires_grad:
            Node._add_grad(
                kernel, Data(d_w.sum(axis=0).reshape(kernel.value.shape), device=device)
            )

        d_cols = w_col.T @ grad_out
        dx_padded = xpv.zeros_like(x_padded)
        for i in range(kh):
            for j in range(kw):
                row = i * kw + j
                dx_padded[
                    :,
                    :,
                    i : i + stride * h_out : stride,
                    j : j + stride * w_out : stride,
                ] += d_cols[:, row * in_ch : (row + 1) * in_ch, :].reshape(
                    batch, in_ch, h_out, w_out
                )

        if x.requires_grad:
            if padding > 0:
                dx = dx_padded[:, :, padding:-padding, padding:-padding]
            else:
                dx = dx_padded
            Node._add_grad(x, Data(dx, device=device))

    out._backward = backward
    return out


def max_pool2d(
    x, kernel_size, stride=None, padding: int = 0, device: Literal["cpu", "gpu"] = "cpu"
):
    from fygrad.node import Node

    x = _ensure_node(x, device)
    xpv = xp(device)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)

    batch, channels, height, width = x.value.shape
    h_out = (height + 2 * padding - kh) // sh + 1
    w_out = (width + 2 * padding - kw) // sw + 1

    if padding > 0:
        pad_val = -xpv.inf
        x_padded = xpv.pad(
            x.value.data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=pad_val,
        )
    else:
        x_padded = x.value.data

    out_val = xpv.zeros((batch, channels, h_out, w_out))
    max_indices = xpv.zeros((batch, channels, h_out, w_out, 2), dtype=xpv.int32)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            w_start = j * sw
            window = x_padded[:, :, h_start : h_start + kh, w_start : w_start + kw]
            flat = window.reshape(batch, channels, -1)
            argmax = xpv.argmax(flat, axis=2)
            out_val[:, :, i, j] = xpv.take_along_axis(
                flat, argmax[..., None], axis=2
            ).squeeze(-1)
            max_indices[:, :, i, j, 0] = argmax // kw
            max_indices[:, :, i, j, 1] = argmax % kw

    out = Node("max_pool2d", Data(out_val, device=device), children=[x], device=device)
    out.requires_grad = x.requires_grad

    def backward():
        if not x.requires_grad:
            return
        dx_padded = xpv.zeros_like(x_padded)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                di = max_indices[:, :, i, j, 0]
                dj = max_indices[:, :, i, j, 1]
                for b in range(batch):
                    for c in range(channels):
                        dx_padded[b, c, h_start + di[b, c], w_start + dj[b, c]] += (
                            out.grad.data[b, c, i, j]
                        )

        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded
        Node._add_grad(x, Data(dx, device=device))

    out._backward = backward
    return out


def avg_pool2d(
    x, kernel_size, stride=None, padding: int = 0, device: Literal["cpu", "gpu"] = "cpu"
):
    from fygrad.node import Node

    x = _ensure_node(x, device)
    xpv = xp(device)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride if stride is not None else kernel_size)

    batch, channels, height, width = x.value.shape
    h_out = (height + 2 * padding - kh) // sh + 1
    w_out = (width + 2 * padding - kw) // sw + 1

    if padding > 0:
        x_padded = xpv.pad(
            x.value.data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        x_padded = x.value.data

    out_val = xpv.zeros((batch, channels, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * sh
            w_start = j * sw
            window = x_padded[:, :, h_start : h_start + kh, w_start : w_start + kw]
            out_val[:, :, i, j] = xpv.mean(window, axis=(2, 3))

    out = Node("avg_pool2d", Data(out_val, device=device), children=[x], device=device)
    out.requires_grad = x.requires_grad

    def backward():
        if not x.requires_grad:
            return
        dx_padded = xpv.zeros_like(x_padded)
        scale = 1.0 / (kh * kw)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                grad = out.grad.data[:, :, i, j][:, :, None, None] * scale
                dx_padded[:, :, h_start : h_start + kh, w_start : w_start + kw] += grad

        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded
        Node._add_grad(x, Data(dx, device=device))

    out._backward = backward
    return out


def flatten(a, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    a = _ensure_node(a, device)
    original_shape = a.value.shape
    value = Data(a.value.data.reshape(original_shape[0], -1), device=device)
    out = Node(f"flatten({a.label})", value, children=[a], device=device)
    out.requires_grad = a.requires_grad

    def backward():
        if a.requires_grad:
            Node._add_grad(
                a, Data(out.grad.data.reshape(original_shape), device=device)
            )

    out._backward = backward
    return out


def mse(values, target, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    values = _ensure_node(values, device)
    target = _ensure_node(target, device)
    xpv = xp(device)
    diff = values.value.data - target.value.data
    loss_value = xpv.mean(diff**2) / 2
    out = Node(
        f"mse({values.label})",
        Data(loss_value, device=device),
        children=[values],
        device=device,
    )
    out.requires_grad = values.requires_grad

    def backward():
        if values.requires_grad:
            grad = out.grad.data * diff / diff.size
            Node._add_grad(values, Data(grad, device=device))

    out._backward = backward
    return out


def cross_entropy(probs, target_indices, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    probs = _ensure_node(probs, device)
    xpv = xp(device)
    batch_size = probs.value.data.shape[0] if probs.value.data.ndim >= 2 else 1
    ps = probs.value.data.reshape(batch_size, -1)
    correct = ps[xpv.arange(batch_size), target_indices]
    loss_value = -xpv.mean(xpv.log(correct + 1e-15))
    out = Node(
        f"cross_entropy({probs.label})",
        Data(loss_value, device=device),
        children=[probs],
        device=device,
    )
    out.requires_grad = probs.requires_grad

    def backward():
        if probs.requires_grad:
            grad = ps.copy()
            grad[xpv.arange(batch_size), target_indices] -= 1.0
            grad = grad.reshape(probs.value.shape) / batch_size
            Node._add_grad(probs, Data(out.grad.data * grad, device=device))

    out._backward = backward
    return out


def binary_cross_entropy(prob, target, device: Literal["cpu", "gpu"] = "cpu"):
    from fygrad.node import Node

    prob = _ensure_node(prob, device)
    target = _ensure_node(target, device)
    xpv = xp(device)
    loss_value = -xpv.mean(
        target.value.data * xpv.log(prob.value.data + 1e-15)
        + (1 - target.value.data) * xpv.log(1 - prob.value.data + 1e-15)
    )
    out = Node(
        f"binary_cross_entropy({prob.label})",
        Data(loss_value, device=device),
        children=[prob],
        device=device,
    )
    out.requires_grad = prob.requires_grad

    def backward():
        if prob.requires_grad:
            grad = (prob.value.data - target.value.data) / (
                (prob.value.data + 1e-15) * (1 - prob.value.data + 1e-15)
            )
            Node._add_grad(prob, Data(out.grad.data * grad, device=device))

    out._backward = backward
    return out
