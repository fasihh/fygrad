import numpy as np
from fygrad.node import Node
from fygrad.optim import Adam
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.arange(7) + 1
y_hat = np.array([9, 8, 10, 12, 11, 13, 14])

b1 = Node("b1", np.random.randn())
b2 = Node("b2", np.random.randn())

optim = Adam([b1, b2], lr=0.01, beta2=0.4)

for i in range(1000):
    y = b1 + b2 * X

    loss = Node.mse(y, y_hat)

    optim.zero_grad()
    loss.backward()
    optim.step()

print(b1, b2)

plt.scatter(X, y_hat, marker="x", c="red")
plt.plot(X, (b1.value + b2.value * X).squeeze())
plt.show()

print(b1.state_dict())