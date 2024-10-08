# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import configparser
from TorchNet.torchnet import Net

# %%
ini = configparser.ConfigParser()
ini.read("config.ini")
batch_size = ini["TORCH"].getint("BATCH_SIZE", 64)
num_epochs = ini["TORCH"].getint("EPOCHS", 10)
save_model = ini["TORCH"].get("SAVE_PYTHON", "net_python.pth")
python_data = ini["TORCH"].get("PYTHON_DATA", "MNIST_data")
alpha = ini["MNIST"].getfloat("ALPHA", 0.1)
hidden_layer_size = ini["MNIST"].getint("HIDDEN_LAYER_SIZE", 10)

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
num_features = int(torch.tensor(training_data.data.shape[1:]).prod().item())
categories = int(training_data.targets.max().item() + 1)

# %%
print(len(train_dataloader))
print(len(test_dataloader))
for X, y in train_dataloader:
    print(
        f"Shape of X [N items in batch, C colors channels, H image height, W image width]: {X.shape}"
    )
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
model = Net(num_features, hidden_layer_size, categories).to(device)
print(model)

# %%
loss_fn = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=alpha)


# %%
def train(
    dataloader: DataLoader,
    model: Net,
    loss_fn: nn.NLLLoss,
    optimizer: torch.optim.Optimizer,
):
    size = len(training_data)
    flatten = torch.nn.Flatten()
    X: torch.Tensor
    y: torch.Tensor
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = flatten.forward(X)

        # Compute prediction error
        pred: torch.Tensor = model(X)
        loss: torch.Tensor = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            train_loss, current = loss.item(), batch * len(X)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(y)
            print(
                f"Accuracy: {(100.0 * correct):>0.1f}% \t loss: {train_loss:>4f}  [{current:>5d}/{size:>5d}]"
            )
            torch.save(model.state_dict(), save_model)
    torch.save(model.state_dict(), save_model)


# %%
def test(dataloader: DataLoader, model: Net, loss_fn: nn.NLLLoss):
    size = len(test_data)
    num_batches = len(dataloader)
    flatten = torch.nn.Flatten()
    test_loss, correct = 0, 0
    X: torch.Tensor
    y: torch.Tensor
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = flatten.forward(X)
            pred: torch.Tensor = model(X)
            loss: torch.Tensor = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    test_loss /= num_batches
    acc = 1.0 * correct / size
    print(
        f"Test: Correct: {correct} \t Accuracy: {acc:>0.4f}% \t Loss: {test_loss:>0.4f} \t [{size}] \n"
    )


# %%
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
torch.save(model.state_dict(), save_model)
print("Done!")

# %%
model = Net(num_features, hidden_layer_size, categories).to(device)
model.load_state_dict(torch.load(save_model))
model.eval()
test(test_dataloader, model, loss_fn)

# %%
