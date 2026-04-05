import numpy as np
import matplotlib.pyplot as plt
from fygrad.node import Node
from fygrad.module import Module, Conv, Linear
from fygrad.data import DataLoader
from fygrad.optim import Adam
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

class ConvNet(Module):
    def __init__(self, device="cpu"):
        super().__init__(device)
        self.conv1 = Conv(1, 8, kernel_size=3, padding=1, device=device)
        self.conv2 = Conv(8, 16, kernel_size=3, padding=1, device=device)
        self.fc    = Linear(16 * 8 * 8, 10, device=device)

    def forward(self, x, return_activations=False):
        conv1 = Node.relu(self.conv1(x), self.device)
        conv2 = Node.relu(self.conv2(conv1), self.device)
        flat = Node.flatten(conv2)
        fc = self.fc(flat)
        if return_activations:
            return conv1, conv2, fc
        return Node.softmax(fc, self.device)

model = ConvNet()

digits = load_digits()
X = digits.data
y = digits.target

ss = StandardScaler()
X_scaled = ss.fit_transform(X).reshape(-1, 1, 8, 8)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# model.load("test/convnet.json")
lr = 0.001
dataloader = DataLoader(X_train, y_train, batch_size=2)
optim = Adam(model.parameters(), lr=lr)

epochs = 100
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

    if i % 20 == 0:
        print(np.round(tot_loss / m, 4).squeeze())
model.save("test/convnet.json")

correct = 0
dataloader = DataLoader(X_test, y_test, batch_size=1)
for X, y in dataloader:
    out = model(X)
    correct += (np.argmax(out.value, axis=1) == y).squeeze()

print("correct =", correct)
print("total =", len(X_test))
print(f"{correct / len(X_test) * 100:.2f}%")

X_sample = X_test[:1]
plt.imshow(ss.inverse_transform(X_sample.reshape(1, -1)).reshape(8, 8), cmap="gray")
conv1_act, conv2_act, out = model(X_sample, return_activations=True)

acts = conv1_act.value[0]
fig, axes = plt.subplots(2,4, figsize=(8,4))

for i, ax in enumerate(axes.flat):
    ax.imshow(acts[i], cmap="viridis")
    ax.set_title(f"Kernel {i}")
    ax.axis("off")

plt.show()
