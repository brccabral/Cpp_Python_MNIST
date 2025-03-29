# %%
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 10), nn.ReLU(), nn.Linear(10, 10), nn.LogSoftmax(1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# %%
# Download training data from open datasets.
training_data = datasets.MNIST(
    root="MNIST_data",
    train=True,
    download=False,
    transform=ToTensor(),
)
size = training_data.data.size(dim=0)

# %%
X_train = training_data.data.reshape(
    (
        training_data.data.size(dim=0),
        training_data.data.size(dim=1) * training_data.data.size(dim=2),
    )
)
X_train.type(torch.float)
X_train = X_train / 255.0
X_train.requires_grad_(True)
Y_train = training_data.targets
print(f"{X_train.shape=}")
print(f"{Y_train.shape=}")


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

model = NeuralNetwork().to(device)
model.train()
print(model)

# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.NLLLoss()

correct_prediction = 0
acc = 0.0

# %%
prediction = torch.Tensor()
for generation in range(500):
    optimizer.zero_grad()
    prediction = model.forward(X_train)
    loss: torch.Tensor = loss_fn(prediction, Y_train)
    loss.backward()
    optimizer.step()
    if generation % 50 == 0:
        correct_prediction = (prediction.argmax(1) == Y_train).type(torch.float).sum().item()
        acc = correct_prediction / size
        print(f"Generation {generation}\t Correct {correct_prediction}\tAccuracy {acc:.4f}\n")

correct_prediction = (prediction.argmax(1) == Y_train).type(torch.float).sum().item()
acc = correct_prediction / size
print(f"Final \t Correct {correct_prediction}\tAccuracy {acc:.4f}\n")


# %%
# Download test data from open datasets.
test_data = datasets.MNIST(
    root="MNIST_data",
    train=False,
    download=False,
    transform=ToTensor(),
)
size = test_data.data.size(dim=0)
X_test = test_data.data.reshape(
    (
        test_data.data.size(dim=0),
        test_data.data.size(dim=1) * test_data.data.size(dim=2),
    )
)
X_test.type(torch.float)
X_test = X_test / 255.0
Y_test = test_data.targets

# %%
model.eval()
prediction = model.forward(X_test)
correct_prediction = (prediction.argmax(1) == Y_test).type(torch.float).sum().item()
acc = correct_prediction / size
print(f"Test \t Correct {correct_prediction}\tAccuracy {acc:.4f}\n")

# %%
