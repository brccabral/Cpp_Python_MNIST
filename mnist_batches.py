# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import configparser

# %%
ini = configparser.ConfigParser()
ini.read("config.ini")
batch_size = int(ini["TORCH"].get("BATCH_SIZE", 64))
num_epochs = int(ini["TORCH"].get("EPOCHS", 10))
save_model = ini["TORCH"].get("SAVE_PYTHON", "net_python.pth")
python_data = ini["TORCH"].get("PYTHON_DATA", "MNIST_data")
alpha = float(ini["MNIST"].get("ALPHA", 0.1))

# %%
# Download training data from open datasets.
training_data = datasets.MNIST(
    root=python_data,
    train=True,
    download=False,
    transform=ToTensor(),
)

# %%
# Download test data from open datasets.
test_data = datasets.MNIST(
    root=python_data,
    train=False,
    download=False,
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %%
num_features = torch.tensor(training_data.data.shape[1:]).prod()
categories = training_data.targets.max().item() + 1

# %%
print(len(train_dataloader))
print(len(test_dataloader))
for X, y in train_dataloader:
    print(f"Shape of X [N items in batch, C colors channels, H image height, W image width]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %%
# Get cpu or gpu device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# %%
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 10), nn.ReLU(), nn.Linear(10, 10), nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# model = NeuralNetwork().to(device)
model = NeuralNetwork2().to(device)
print(model)

# %%
loss_fn = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=alpha)


# %%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(y)
            print(
                f"Accuracy: {(100*correct):>0.1f}% \t loss: {loss:>4f}  [{current:>5d}/{size:>5d}]"
            )
            torch.save(model.state_dict(), save_model)


# %%
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test: Accuracy: {(100*correct):>0.1f}% \t Avg loss: {test_loss:>4f} \n")


# %%
epochs = num_epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

# %%
model = NeuralNetwork2().to(device)
model.load_state_dict(torch.load(save_model))
test(test_dataloader, model, loss_fn)

# %%
