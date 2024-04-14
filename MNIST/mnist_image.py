from PIL import Image


class MNIST_Image:
    def __init__(
        self,
        rows: int,
        cols: int,
        label: int,
        pixels: bytes,
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

    def get_pixels_as_float_list(self) -> list[float]:
        return [self._label] + list(map(float, self._pixels))
