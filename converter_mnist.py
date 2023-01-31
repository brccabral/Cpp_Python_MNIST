import configparser
import time
import numpy as np
from PIL import Image

TRAIN_IMAGE_MAGIC = 2051
TRAIN_LABEL_MAGIC = 2049
TEST_IMAGE_MAGIC = 2051
TEST_LABEL_MAGIC = 2049


class MNIST_Image:
    def __init__(
        self,
        rows: int,
        cols: int,
        label: int,
        pixels: list[bytes],
        item_id: int,
    ):
        self._rows = rows
        self._cols = cols
        self._label = label
        self._pixels = pixels
        self._db_item_id = item_id

    def save_as_png(self, save_dir: str):
        img = Image.frombytes("L", (self._rows, self._cols), self._pixels)
        img.save(f"{save_dir}/{self._db_item_id}_{self._label}.png")

    def save_as_csv(self, save_filename):
        if self._db_item_id == 0:
            outfile = open(f"{save_filename}", "w")
        else:
            outfile = open(f"{save_filename}", "a")

        outfile.write(str(self._label))
        outfile.write(",")
        outfile.write(",".join(list(map(str, self._pixels))))
        outfile.write("\n")
        outfile.close()

    def get_pixels_as_int_list(self) -> list[float]:
        return [self._label] + list(map(float, self._pixels))


class MNIST_Dataset:
    def __init__(self, image_filename, label_filename, image_magic, label_magic):
        self._images: list[MNIST_Image] = []
        self._image_filename = image_filename
        self._label_filename = label_filename
        self._image_magic = image_magic
        self._label_magic = label_magic

    def save_dataset_as_png(self, save_dir: str):
        for img in self._images:
            img.save_as_png(save_dir)

    def save_dataset_as_csv(self, save_dir: str):
        for img in self._images:
            img.save_as_csv(save_dir)

    def read_mnist_db(self, max_items: int) -> list[MNIST_Image]:
        image_file = open(self._image_filename, "rb")
        if not image_file:
            raise Exception("Failed open image file")

        label_file = open(self._label_filename, "rb")
        if not label_file:
            raise Exception("Failed open label file")

        magic = int.from_bytes(image_file.read(4), "big")
        if magic != self._image_magic:
            raise Exception(f"Incorrect image file magic {magic}")

        num_items = int.from_bytes(image_file.read(4), "big")
        rows = int.from_bytes(image_file.read(4), "big")
        cols = int.from_bytes(image_file.read(4), "big")

        magic = int.from_bytes(label_file.read(4), "big")
        if magic != self._label_magic:
            raise Exception(f"Incorrect image file magic {magic}")

        num_labels = int.from_bytes(label_file.read(4), "big")

        if num_items != num_labels:
            raise Exception("image file nums should equal to label num")

        n_items = max_items
        if max_items > num_items:
            n_items = num_items

        for item_id in range(n_items):
            pixels = image_file.read(rows * cols)
            label = ord(label_file.read(1))

            m_image = MNIST_Image(rows, cols, label, pixels, item_id)

            self._images.append(m_image)

        image_file.close()
        label_file.close()

    def to_numpy(self) -> np.ndarray:
        return np.asarray([img.get_pixels_as_int_list() for img in self._images])


class NeuralNet:
    def __init__(self):
        pass


def init_params(
    hidden_layer_size: int, categories: int, num_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = np.random.rand(hidden_layer_size, num_features) - 0.5
    b1 = np.random.rand(hidden_layer_size, 1) - 0.5
    W2 = np.random.rand(categories, hidden_layer_size) - 0.5
    b2 = np.random.rand(categories, 1) - 0.5
    return (W1, b1, W2, b2)


def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)


def softmax(Z: np.array) -> np.array:
    return np.exp(Z) / sum(np.exp(Z))


def forward_prop(
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1: np.ndarray = W1.dot(X) + b1  # W1 10,784 ||| X 784,60000 ||| W.X 10,60000
    A1: np.ndarray = ReLU(Z1)
    Z2: np.ndarray = W2.dot(A1) + b2
    A2: np.ndarray = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # create matrix with correct size
    # np.arange(Y.size) - will return a list from 0 to Y.size
    # Y - will contain the column index to set the value of 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T  # transpose


def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    return Z > 0


def back_prop(
    Z1: np.ndarray,
    A1: np.ndarray,
    Z2: np.ndarray,
    A2: np.ndarray,
    W2: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    one_hot_Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = Y.size
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1: np.ndarray = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    dW1: np.ndarray,
    db1: np.ndarray,
    dW2: np.ndarray,
    db2: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2: np.ndarray) -> np.ndarray:
    return np.argmax(A2, 0)


def get_correct_prediction(predictions: np.ndarray, Y: np.ndarray) -> float:
    return np.sum(predictions == Y)


def get_accuracy(correct_prediction: int, size: int):
    return 1.0 * correct_prediction / size


def main():

    np.random.seed(int(time.time()))

    ini = configparser.ConfigParser()
    ini.read("config.ini")

    num_generations = int(ini["MNIST"].get("GENERATIONS", 5))
    max_items = int(ini["MNIST"].get("MAX_ITEMS", 5))
    save_img = bool(ini["MNIST"].get("SAVE_IMG", 5))
    alpha = float(ini["MNIST"].get("ALPHA", 5))
    hidden_layer_size = int(ini["MNIST"].get("HIDDEN_LAYER_SIZE", 10))

    base_dir = ini["MNIST"].get("BASE_DIR", "MNIST")
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

    mat = train_dataset.to_numpy()

    X_train = mat[:, 1:]
    Y_train: np.ndarray = mat[:, 0]
    X_train /= 255.0
    Y_train = Y_train.astype(int)

    categories = np.max(Y_train) + 1

    X = X_train.T

    W1, b1, W2, b2 = init_params(hidden_layer_size, categories, X_train.shape[1])
    one_hot_Y = one_hot(Y_train)

    correct_prediction = 0
    acc = 0.0

    for generation in range(num_generations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y_train, one_hot_Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if generation % 50 == 0:
            predictions = get_predictions(A2)
            correct_prediction = get_correct_prediction(predictions, Y_train)
            acc = get_accuracy(correct_prediction, Y_train.size)
            print(
                f"Generation: {generation}\tCorrect {correct_prediction}\tAccuracy: {acc}"
            )
    print(
        f"Final\tCorrect {correct_prediction}\tAccuracy: {get_accuracy(get_correct_prediction(get_predictions(A2), Y_train), Y_train.size)}"
    )

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

    if save_img:
        test_dataset.save_dataset_as_png(save_dir)

    test_dataset.save_dataset_as_csv(save_dir + "/test.csv")

    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

    predictions = get_predictions(A2)
    correct_prediction = get_correct_prediction(predictions, Y_train)
    acc = get_accuracy(correct_prediction, Y_train.size)
    print(f"Test: {generation}\tCorrect {correct_prediction}\tAccuracy: {acc}")


if __name__ == "__main__":
    main()
