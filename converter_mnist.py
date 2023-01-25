import sys
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


def read_mnist_db(
    img_path: str, label_path: str, max_items: int, save_dir: str, save_img: bool
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
        print(item_id)
        pixels = image_file.read(rows * cols)
        label = ord(label_file.read(1))

        m_image = MNIST_Image(rows, cols, label, pixels, item_id)

        if save_img:
            m_image.save_as_png(save_dir)

        m_image.save_as_csv(save_dir)

        dataset.append(m_image)

    image_file.close()
    label_file.close()
    return dataset


def get_pixels_as_int_list(image: MNIST_Image):
    return list(map(int, image.pixels))


def to_numpy(dataset: list[MNIST_Image]):
    return np.asarray(list(map(get_pixels_as_int_list, dataset)))


def main(argc: int, argv: list[str]):

    np.random.seed(int(time.time()))

    num_generations = int(argv[1])
    max_items = int(argv[2])
    save_img = int(argv[3])
    alpha = float(argv[4])

    base_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST"
    save_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST/train"
    img_path = base_dir + "/train-images.idx3-ubyte"
    label_path = base_dir + "/train-labels.idx1-ubyte"

    dataset = read_mnist_db(img_path, label_path, max_items, save_dir, save_img)
    mat = to_numpy(dataset)
    print(mat.shape)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(len(sys.argv))
        print("Wrong parameters: converter_mnist GENERATIONS MAX_ITEMS SAVE_IMG ALPHA")
        exit(1)
    main(len(sys.argv), sys.argv)
