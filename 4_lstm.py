import numpy as np
from fygrad.node import Device, Node
from fygrad.module import Module, LSTM
from fygrad.data import DataLoader
from fygrad.optim import SGD
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(69727)

class MultiLayerLSTM(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        device: Device = "cpu",
    ):
        super().__init__(device)

        self.output_size = output_size

        sizes = [input_size] + [hidden_size] * num_layers

        self.layers = [
            LSTM(sizes[i], sizes[i+1], device=device)
            for i in range(num_layers)
        ]

        for i, layer in enumerate(self.layers):
            self.add_module(f"rnn{i}", layer)

        self.Wy = Node.randn("Wy", (hidden_size, output_size), device=device)
        self.by = Node.zeros("by", (1, output_size), device=device)


    def forward(self, seq):
        if not isinstance(seq, list):
            seq = [
                Node(f"x{t}", seq[:, t, :], device=self.device)
                for t in range(seq.shape[1])
            ]

        for layer in self.layers:
            seq = layer(seq)
        h_last = seq[-1]
        return Node.softmax(Node.matmul(h_last, self.Wy) + self.by)        


model = MultiLayerLSTM(8, 10, 10, 2)

digits = load_digits()
X = digits.data[:10]
y = digits.target[:10]

ss = StandardScaler()
X_scaled = ss.fit_transform(X).reshape(-1, 8, 8)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

lr = 0.15
dataloader = DataLoader(X_train, y_train, batch_size=2)
optim = SGD(model.parameters(), lr=lr)

epochs = 2
for i in range(epochs):
    tot_loss = 0
    m = 0
    for X, y in dataloader:
        optim.zero_grad()
        out = model(X)
        loss = Node.cross_entropy(out, y)
        tot_loss += loss.value
        m += 1
        loss.backward()
        optim.step()

    print(np.round(tot_loss / m, 4).squeeze())

correct = 0
dataloader = DataLoader(X_test, y_test, batch_size=1)
for X, y in dataloader:
    out = model(X)
    correct += (np.argmax(out.value, axis=1) == y).squeeze()

print(f"{correct / len(X_test) * 100:.2f}%")
