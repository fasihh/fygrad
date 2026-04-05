import numpy as np
from fygrad.node import Node
from fygrad.module import RNN
from fygrad.optim import SGD

np.random.seed(42)

dataset = [
    # sample 1
    (
        [
            [0.2, 1.1, 0.3, 0.7, 0.5],
            # [0.9, 0.1, 0.4, 0.2, 0.8],
            # [0.3, 0.6, 0.7, 0.1, 0.2],
            # [0.5, 0.4, 0.2, 0.9, 0.3],
        ],
        1,
    ),
    # sample 2
    (
        [
            [0.7, 0.3, 0.8, 0.2, 0.1],
            [0.4, 0.9, 0.5, 0.6, 0.3],
            [0.2, 0.1, 0.9, 0.7, 0.4],
            [0.6, 0.8, 0.2, 0.3, 0.5],
        ],
        0,
    ),
    # sample 3
    (
        [
            [0.1, 0.5, 0.2, 0.8, 0.9],
            [0.3, 0.6, 0.4, 0.1, 0.7],
            [0.8, 0.2, 0.3, 0.5, 0.6],
            [0.4, 0.7, 0.1, 0.9, 0.2],
        ],
        1,
    ),
]

flag = True

lr = 0.01
model = RNN(input_size=5, hidden_size=8, output_size=1)
op = SGD(model.parameters(), lr=lr)
for epoch in range(100):
    tot_loss = 0
    for X, y in dataset:
        prob = model(X)
        loss = Node.binary_cross_entropy(prob, y)
        if flag:
            print(loss)
            flag = False

        tot_loss += loss.value.squeeze()

        op.zero_grad()
        loss.backward()
        op.step()
    
    if epoch % 10 == 0:
        print("Loss:", np.round(tot_loss / len(dataset), 4))
