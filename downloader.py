from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="./MNIST_data",
    train=True,
    download=True,
    transform=ToTensor(),
)
