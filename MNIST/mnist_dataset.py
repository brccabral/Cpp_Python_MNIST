from .mnist_image import MNIST_Image
import numpy as np


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
            label = int.from_bytes(label_file.read(1), "big")

            m_image = MNIST_Image(rows, cols, label, pixels, item_id)

            self._images.append(m_image)

        image_file.close()
        label_file.close()

    def to_numpy(self) -> np.ndarray:
        return np.asarray([img.get_pixels_as_float_list() for img in self._images])
