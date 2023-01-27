import configparser
import time
import numpy as np
from PIL import Image


class MNIST_Image:
    def __init__(
        self, rows: int, cols: int, label: int, pixels: list[bytes], item_id: int
    ):
        self._rows = rows
        self._cols = cols
        self._label = label
        self.pixels = pixels
        self._db_item_id = item_id

    def save_as_png(self, save_dir: str):
        img = Image.frombytes("L", (self._rows, self._cols), self.pixels)
        img.save(f"{save_dir}/{self._db_item_id}_{self._label}.png")

    def save_as_csv(self, save_dir: str):
        if self._db_item_id == 0:
            outfile = open(f"{save_dir}/python_csv.txt", "w")
        else:
            outfile = open(f"{save_dir}/python_csv.txt", "a")

        outfile.write(str(self._label))
        outfile.write(",")
        outfile.write(",".join(list(map(str, self.pixels))))
        outfile.write("\n")
        outfile.close()


def save_dataset_as_png(dataset: list[MNIST_Image], save_dir: str):
    for img in dataset:
        img.save_as_png(save_dir)


def read_mnist_db(
    img_path: str, label_path: str, max_items: int, save_dir: str
) -> list[MNIST_Image]:
    dataset: list[MNIST_Image] = []

    image_file = open(img_path, "rb")
    if not image_file:
        print("Failed open image file")
        return dataset

    label_file = open(label_path, "rb")
    if not label_file:
        print("Failed open label file")
        return dataset

    magic = int.from_bytes(image_file.read(4), "big")
    if magic != 2051:
        print(f"Incorrect image file magic {magic}")
        return dataset

    num_items = int.from_bytes(image_file.read(4), "big")
    rows = int.from_bytes(image_file.read(4), "big")
    cols = int.from_bytes(image_file.read(4), "big")

    magic = int.from_bytes(label_file.read(4), "big")
    if magic != 2049:
        print(f"Incorrect image file magic {magic}")
        return dataset

    num_labels = int.from_bytes(label_file.read(4), "big")

    if num_items != num_labels:
        print("image file nums should equal to label num")
        return dataset

    n_items = max_items
    if max_items > num_items:
        n_items = num_items

    for item_id in range(n_items):
        pixels = image_file.read(rows * cols)
        label = ord(label_file.read(1))

        m_image = MNIST_Image(rows, cols, label, pixels, item_id)

        m_image.save_as_csv(save_dir)

        dataset.append(m_image)

    image_file.close()
    label_file.close()
    return dataset


def get_pixels_as_int_list(image: MNIST_Image) -> list[float]:
    return [image._label] + list(map(float, image.pixels))


def to_numpy(dataset: list[MNIST_Image]) -> np.ndarray:
    return np.asarray(list(map(get_pixels_as_int_list, dataset)))


def init_params(
    categories: int, num_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = np.random.rand(categories, num_features) - 0.5
    b1 = np.random.rand(categories, 1) - 0.5
    W2 = np.random.rand(categories, categories) - 0.5
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

    base_dir = ini["MNIST"].get("BASE_DIR", "MNIST")
    save_dir = base_dir + "/train"
    img_filename = ini["MNIST"].get("TRAIN_IMAGE_FILE", "train-images.idx3-ubyte")
    img_path = base_dir + "/" + img_filename
    label_filename = ini["MNIST"].get("TRAIN_LABEL_FILE", "train-labels.idx1-ubyte")
    label_path = base_dir + "/" + label_filename

    dataset = read_mnist_db(img_path, label_path, max_items, save_dir)

    if save_img:
        save_dataset_as_png(dataset, save_dir)

    mat = to_numpy(dataset)

    X_train = mat[:, 1:]
    Y_train: np.ndarray = mat[:, 0]
    X_train /= 255.0
    Y_train = Y_train.astype(int)

    categories = np.max(Y_train) + 1

    X = X_train.T

    W1, b1, W2, b2 = init_params(categories, X_train.shape[1])
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


if __name__ == "__main__":
    main()
