# fygrad

A tiny autograd and neural-network toolkit with NumPy/CuPy support.

## Install

```bash
pip install fygrad
```

## Quick start

```python
from fygrad import Node

x = Node("x", [[1.0, 2.0], [3.0, 4.0]])
y = F.sum(x * 2)
y.backward()
print(x.grad)
```

## Notes

- GPU support uses CuPy if it is available.
- The API is intentionally minimal.

## License

MIT
