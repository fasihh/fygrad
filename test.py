from module import Module, Linear
from node import Node
from data import DataLoader, Dataset
from backend import gpu_available
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

digits = load_digits()

X = digits.data
y = digits.target

ss = StandardScaler()
X_scaled = ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

dataset_train = Dataset(X_train, y_train)
DEVICE = "gpu" if gpu_available() else "cpu"
dataloader_train = DataLoader(dataset_train, device=DEVICE)

class MLP(Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.fc1 = Linear(input_dim, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 10)
    
    def forward(self, x: np.ndarray):
        x = self.fc1(x)
        x = Node.sigmoid(x)
        x = self.fc2(x)
        x = Node.sigmoid(x)
        x = self.fc3(x)
        return Node.softmax(x)

model = MLP(64)
if DEVICE == "gpu":
    model.to_gpu()

epochs = 100
batch_size = 32
lr = 0.5
lmbda = 0.001
for i in range(epochs):
    idx = np.random.permutation(len(X_train))
    X_shuf = X_train[idx]
    y_shuf = y_train[idx]

    tot_loss = 0
    m = 0
    for X, y in dataloader_train:
        probs = model(X)
        loss = Node.cross_entropy(probs, y)

        params = model.parameters()
        for param in params:
            param.zero_grad()

        # L1 reg
        reg = sum(
            (w.abs().sum() for w in params),
            start=Node("reg", 0.0, device=DEVICE),
        )

        cost = loss + reg * lmbda

        tot_loss += loss.value
        
        cost.backward()

        for param in params:
            param.value -= lr * param.grad

        m += 1
    
    if i % 10 == 0:
        print(f"#{i // 10} Loss:", np.round((tot_loss / m).squeeze(), 4))


probs = model(dataloader_train.xp.asarray(X_test))
correct = np.sum(probs.value.argmax(axis=1) == y_test)
total = len(probs)
print("Correct:", correct)
print("Total:", total)
print("Test Accuracy:", np.round(correct / total, 4) * 100)
