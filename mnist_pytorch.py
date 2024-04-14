import configparser
import time
import numpy as np
from MNIST.mnist_dataset import MNIST_Dataset
from TorchNet.torchnet import Net
import torch

TRAIN_IMAGE_MAGIC = 2051
TRAIN_LABEL_MAGIC = 2049
TEST_IMAGE_MAGIC = 2051
TEST_LABEL_MAGIC = 2049


def main():
    np.random.seed(int(time.time()))

    ini = configparser.ConfigParser()
    ini.read("config.ini")

    num_generations = ini["MNIST"].getint("GENERATIONS", 5)
    max_items = ini["MNIST"].getint("MAX_ITEMS", 5)
    save_img = ini["MNIST"].getboolean("SAVE_IMG", 0)
    alpha = ini["MNIST"].getfloat("ALPHA", 0.1)
    hidden_layer_size = ini["MNIST"].getint("HIDDEN_LAYER_SIZE", 10)

    base_dir = ini["MNIST"].get("BASE_DIR", "MNIST_data/MNIST/raw")

    save_dir = base_dir + "/train"
    img_filename = ini["MNIST"].get("TRAIN_IMAGE_FILE", "train-images.idx3-ubyte")
    img_path = base_dir + "/" + img_filename
    label_filename = ini["MNIST"].get("TRAIN_LABEL_FILE", "train-labels.idx1-ubyte")
    label_path = base_dir + "/" + label_filename

    train_dataset = MNIST_Dataset(
        img_path,
        label_path,
        TRAIN_IMAGE_MAGIC,
        TRAIN_LABEL_MAGIC,
    )
    train_dataset.read_mnist_db(max_items)

    if save_img:
        train_dataset.save_dataset_as_png(save_dir)

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv")

    train_mat = train_dataset.to_numpy()

    Y_train = MNIST_Dataset.get_Y(train_mat)
    X_train = MNIST_Dataset.get_X(train_mat)
    X_train /= 255.0
    Y_train = Y_train.astype(int)

    categories = int(np.max(Y_train) + 1)

    X_tensor_train = torch.tensor(X_train)
    X_tensor_train.requires_grad_(True)
    X_tensor_train = X_tensor_train.type(torch.float32)

    Y_tensor_train = torch.tensor(Y_train)
    # Y_tensor_train = Y_tensor_train.type(torch.long)

    print(f"{X_tensor_train.shape=}")
    print(f"{Y_tensor_train.shape=}")

    neural_net = Net(X_train.shape[1], hidden_layer_size, categories)
    neural_net.train()

    optimizer = torch.optim.SGD(neural_net.parameters(), lr=alpha)
    loss_fn = torch.nn.NLLLoss()

    correct_prediction = 0
    acc = 0.0

    for generation in range(num_generations):
        optimizer.zero_grad()
        output = neural_net.forward(X_tensor_train)
        loss = loss_fn.forward(output, Y_tensor_train)
        loss.backward()
        optimizer.step()

        if generation % 50 == 0:
            predictions = output.argmax(1) == Y_tensor_train
            correct_prediction = predictions.type(torch.int).sum().item()
            acc = 1.0 * correct_prediction / Y_train.size
            print(
                f"Generation: {generation}\tCorrect {correct_prediction}\tAccuracy: {acc:.4f}"
            )

    output = neural_net.forward(X_tensor_train)
    predictions = output.argmax(1) == Y_tensor_train
    correct_prediction = predictions.type(torch.int).sum().item()
    acc = 1.0 * correct_prediction / Y_train.size
    print(f"Final\tCorrect {correct_prediction}\tAccuracy: {acc:.4f}")

    save_dir = base_dir + "/test"
    img_filename = ini["MNIST"].get("TEST_IMAGE_FILE", "t10k-images.idx3-ubyte")
    img_path = base_dir + "/" + img_filename
    label_filename = ini["MNIST"].get("TEST_LABEL_FILE", "t10k-labels.idx1-ubyte")
    label_path = base_dir + "/" + label_filename

    test_dataset = MNIST_Dataset(
        img_path,
        label_path,
        TEST_IMAGE_MAGIC,
        TEST_LABEL_MAGIC,
    )
    test_dataset.read_mnist_db(max_items)

    if save_img:
        test_dataset.save_dataset_as_png(save_dir)

    test_dataset.save_dataset_as_csv(save_dir + "/test.csv")

    test_mat = test_dataset.to_numpy()

    Y_test = MNIST_Dataset.get_Y(test_mat)
    X_test = MNIST_Dataset.get_X(test_mat)
    X_test /= 255.0
    Y_test = Y_test.astype(int)

    X_tensor_test = torch.tensor(X_test)
    X_tensor_test = X_tensor_test.type(torch.float32)
    Y_tensor_test = torch.tensor(Y_test)
    # Y_tensor_train = Y_tensor_train.type(torch.long)

    output = neural_net.forward(X_tensor_test)
    predictions = output.argmax(1) == Y_tensor_test
    correct_prediction = predictions.type(torch.int).sum().item()
    acc = 1.0 * correct_prediction / Y_test.size
    print(f"Test\tCorrect {correct_prediction}\tAccuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
