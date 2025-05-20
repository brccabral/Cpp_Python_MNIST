#include <SimpleIni/SimpleIni.h>
#include <iostream>
#include <MNIST/MNIST_Dataset.hpp>
#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

CNdArray to_matrix(const CNumpy &np, const std::vector<MNIST_Image> &_images)
{
    const int number_images = _images.size();
    const int number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    const npy_intp dims[] = {number_images, number_pixels + 1};
    auto mat = np.ndarray(2, dims, NPY_FLOAT);
    for (int img = 0; img < number_images; img++)
    {
        mat(img, 0) = float(_images.at(img)._label);
        for (int pix = 0; pix < number_pixels; pix++)
        {
            mat(img, pix + 1) = (unsigned char) _images.at(img)._pixels[pix];
        }
    }
    return mat;
}

CNdArray get_Y(const CNumpy &np, const CNdArray &mat)
{
    const npy_intp rows = mat.rows();
    const npy_intp dims[] = {rows, 1};
    const int ndtype = mat.ndtype;
    auto Y = np.ndarray(2, dims, ndtype);
    for (npy_intp r = 0; r < rows; ++r)
    {
        Y(r, 0) = mat(r, 0);
    }
    return Y;
}

CNdArray get_X(const CNumpy &np, const CNdArray &mat)
{
    const npy_intp rows = mat.rows();
    const npy_intp cols = mat.cols();
    const npy_intp dims[] = {rows, cols - 1};
    const int ndtype = mat.ndtype;
    auto X = np.ndarray(2, dims, ndtype);
    for (npy_intp r = 0; r < rows; ++r)
    {
        for (npy_intp c = 1; c < cols; ++c)
        {
            X(r, c - 1) = mat(r, c);
        }
    }
    return X;
}

int main()
{
    std::srand((unsigned int) time(nullptr)); // NOLINT(*-msc51-cpp)

    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    }
    SI_ASSERT(rc == SI_OK);

    auto num_generations = (int) ini.GetLongValue("MNIST", "GENERATIONS", 5);
    auto max_items = (int) ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    auto alpha = (float) ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    auto hidden_layer_size = (int) ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename =
            ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

    std::cout << PRINT_VAR(num_generations) << " " << PRINT_VAR(max_items) << " "
              << PRINT_VAR(save_img) << " " << PRINT_VAR(alpha) << " "
              << PRINT_VAR(hidden_layer_size) << " " << PRINT_VAR(base_dir) << " "
              << PRINT_VAR(save_dir) << " " << PRINT_VAR(img_filename) << " " << PRINT_VAR(img_path)
              << " " << PRINT_VAR(label_filename) << " " << PRINT_VAR(label_path) << " "
              << std::endl;

    MNIST_Dataset train_dataset(
            img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);
    std::cout << PRINT_VAR(train_dataset.get_images_length()) << std::endl;
    std::cout << PRINT_VAR(train_dataset.get_label_from_index(4)) << std::endl;

#ifdef CV_SAVE_IMAGES
    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);
    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");
#endif // CV_SAVE_IMAGES

    const auto &np = CNumpy::instance();
    auto train_mat = to_matrix(np, train_dataset._images);

    auto Y_train = get_Y(np, train_mat);
    auto X_train = get_X(np, train_mat);

    std::cout << Y_train(4, 0) << std::endl;
    std::cout << X_train.rows() << "," << X_train.cols() << std::endl;
    for (int c = 0; c < X_train.cols(); c++)
        std::cout << X_train(4, c) << ", ";
    std::cout << std::endl;

    int categories = np.max(Y_train) + 1;
    std::cout << categories << std::endl;

    X_train /= 255.0f;
    for (int c = 0; c < X_train.cols(); c++)
        std::cout << X_train(4, c) << ", ";
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
