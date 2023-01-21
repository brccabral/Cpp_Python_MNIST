# converter_mnist.py
# raw binary MNIST to Keras text file

# target format:
# 0 0 1 0 0 0 0 0 0 0 ** 0 0 152 27 .. 0
# 0 1 0 0 0 0 0 0 0 0 ** 0 0 38 122 .. 0
#   10 vals at [0-9]    784 vals at [11-795]
# dummy ** seperator at [10]


from PIL import Image


def generate(img_bin_file, lbl_bin_file, result_file, save_location, n_images):

    img_bf = open(img_bin_file, "rb")  # binary image pixels
    lbl_bf = open(lbl_bin_file, "rb")  # binary labels
    res_tf = open(result_file, "w")  # result text file

    # header info are in 'big endian', used in old CPUs (bytes are in reverse order)
    img_magic = int.from_bytes(img_bf.read(4), "big")  # check if 2051
    number_of_images = int.from_bytes(img_bf.read(4), "big")
    img_rows = int.from_bytes(img_bf.read(4), "big")
    img_cols = int.from_bytes(img_bf.read(4), "big")
    print(
        f"Image Magic {img_magic} Number of images {number_of_images} Image Size {img_rows}x{img_cols}"
    )

    label_magic = int.from_bytes(lbl_bf.read(4), "big")  # check if 2049
    number_of_items = int.from_bytes(lbl_bf.read(4), "big")
    print(f"Label Magic {label_magic} Number of items {number_of_items}")

    for image in range(n_images):  # number images requested
        # digit label first
        lbl = ord(lbl_bf.read(1))  # get label like '3' (one byte)
        encoded = [0] * 10  # make one-hot vector
        encoded[lbl] = 1
        for encode in range(10):
            res_tf.write(str(encoded[encode]))
            res_tf.write(" ")  # like 0 0 0 1 0 0 0 0 0 0

        res_tf.write("** ")  # arbitrary for readibility

        img_bytes = img_bf.read(img_rows * img_cols)
        img = Image.frombytes("L", (img_rows, img_cols), img_bytes)
        img.save(f"{save_location}/{image}_{lbl}.png")

        # now do the image pixels
        for value in range(img_rows * img_cols):  # get 784 vals for each image file
            val = img_bytes[value]
            res_tf.write(str(val))
            if value != 783:
                res_tf.write(" ")  # avoid trailing space
        res_tf.write("\n")  # next image

    img_bf.close()
    lbl_bf.close()
    # close the binary files
    res_tf.close()  # close the result text file


def main():
    # change target file names, uncomment as necessary

    # make training data
    generate(
        "./MNIST/train-images.idx3-ubyte",
        "./MNIST/train-labels.idx1-ubyte",
        "./MNIST/mnist_train_keras.txt",
        "./MNIST/train",
        n_images=10,
    )  # first n images

    # make test data
    generate(
        "./MNIST/t10k-images.idx3-ubyte",
        "./MNIST/t10k-labels.idx1-ubyte",
        "./MNIST/mnist_test_keras.txt",
        "./MNIST/test",
        n_images=5,
    )  # first n images


if __name__ == "__main__":
    main()
