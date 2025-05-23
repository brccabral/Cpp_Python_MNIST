import configparser
import time
import numpy as np
from MNIST.mnist_dataset import MNIST_Dataset
from NeuralNet.neuralnet import NeuralNet

TRAIN_IMAGE_MAGIC = 2051
TRAIN_LABEL_MAGIC = 2049
TEST_IMAGE_MAGIC = 2051
TEST_LABEL_MAGIC = 2049


def main():
    np.random.seed(int(time.time()))

    ini = configparser.ConfigParser()
    ini.read("config.ini")

    num_generations = ini["MNIST"].getint("GENERATIONS", 5)
    max_items = ini["MNIST"].getint("MAX_ITEMS", 10)
    save_img = ini["MNIST"].getboolean("SAVE_IMG", 0)
    alpha = ini["MNIST"].getfloat("ALPHA", 0.1)
    hidden_layer_size = ini["MNIST"].getint("HIDDEN_LAYER_SIZE", 10)

    base_dir = ini["MNIST"].get("BASE_DIR", "MNIST_data/MNIST/raw")
    save_dir = base_dir + "/train"
    img_filename = ini["MNIST"].get("TRAIN_IMAGE_FILE", "train-images-idx3-ubyte")
    img_path = base_dir + "/" + img_filename
    label_filename = ini["MNIST"].get("TRAIN_LABEL_FILE", "train-labels-idx1-ubyte")
    label_path = base_dir + "/" + label_filename

    print(
        f"{num_generations=} {max_items=} {save_img=} {alpha=} {hidden_layer_size=} {base_dir=} {save_dir=} {img_filename=} {img_path=} {label_filename=} {label_path=}"
    )

    train_dataset = MNIST_Dataset(
        img_path,
        label_path,
        TRAIN_IMAGE_MAGIC,
        TRAIN_LABEL_MAGIC,
    )
    train_dataset.read_mnist_db(max_items)
    print(f"{len(train_dataset._images)=}")
    print(f"{train_dataset._images[4]._label=}")

    if save_img:
        train_dataset.save_dataset_as_png(save_dir)

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv")

    train_mat = train_dataset.to_numpy()

    Y_train = MNIST_Dataset.get_Y(train_mat)
    X_train = MNIST_Dataset.get_X(train_mat)
    X_train /= 255.0
    print(f"{Y_train[4]=}")
    print(f"{X_train.shape=}")
    print(f"{X_train[4, :]=}")

    Y_train = Y_train.astype(int)

    categories = np.max(Y_train) + 1

    X_train_T = X_train.T

    neural_net = NeuralNet(X_train.shape[1], hidden_layer_size, categories)
    one_hot_Y = NeuralNet.one_hot(Y_train)

    for generation in range(num_generations):
        output = neural_net.forward_prop(X_train_T)

        if generation % 50 == 0:
            predictions = NeuralNet.get_predictions(output)
            correct_prediction = NeuralNet.get_correct_prediction(predictions, Y_train)
            acc = NeuralNet.get_accuracy(correct_prediction, Y_train.size)
            print(f"Generation: {generation}\tCorrect {correct_prediction}\tAccuracy: {acc:.4f}")

        neural_net.back_prop(X_train_T, Y_train, one_hot_Y, alpha)

    output = neural_net.forward_prop(X_train_T)
    predictions = NeuralNet.get_predictions(output)
    correct_prediction = NeuralNet.get_correct_prediction(predictions, Y_train)
    acc = NeuralNet.get_accuracy(correct_prediction, Y_train.size)

    print(f"Final\tCorrect {correct_prediction}\tAccuracy: {acc:.4f}")

    save_dir = base_dir + "/test"
    img_filename = ini["MNIST"].get("TEST_IMAGE_FILE", "t10k-images-idx3-ubyte")
    img_path = base_dir + "/" + img_filename
    label_filename = ini["MNIST"].get("TEST_LABEL_FILE", "t10k-labels-idx1-ubyte")
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

    X_test_T = X_test.T

    output = neural_net.forward_prop(X_test_T)

    predictions = NeuralNet.get_predictions(output)
    correct_prediction = NeuralNet.get_correct_prediction(predictions, Y_test)
    acc = NeuralNet.get_accuracy(correct_prediction, Y_test.size)
    print(f"Test: \tCorrect {correct_prediction}\tAccuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
