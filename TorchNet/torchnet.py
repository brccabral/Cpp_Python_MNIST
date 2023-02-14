from torch import nn


class Net(nn.Module):
    def __init__(self, num_features: int, hidden_layer_size: int, categories: int):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features, hidden_layer_size)
        self.out = nn.Linear(hidden_layer_size, categories)
        self.relu = nn.ReLU()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(num_features, hidden_layer_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_size, categories),
        #     nn.LogSoftmax(dim=1),
        # )

    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        x = self.fc1.forward(x)
        x = self.relu(x)
        x = nn.LogSoftmax(dim=1).forward(self.out.forward(x))
        return x
