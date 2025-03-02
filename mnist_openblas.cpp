#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <cblas.h>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

typedef struct MatrixDouble
{
    uint rows;
    uint cols;
    double *data;
} MatrixDouble;

MatrixDouble *to_openblas(const std::vector<MNIST_Image> &_images)
{
    const int number_images = _images.size();
    const int number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    auto *mat = (MatrixDouble *) malloc(sizeof(MatrixDouble) * number_images * (number_pixels + 1));
    mat->rows = number_images;
    mat->cols = number_pixels;
    for (int img = 0; img < number_images; img++)
    {
        mat->data[img * number_images + 0] = float(_images.at(img)._label);
        for (int pix = 0; pix < number_pixels; pix++)
        {
            mat->data[img * number_images + pix + 1] = (unsigned char) _images.at(img)._pixels[pix];
        }
    }
    return mat;
}

MatrixDouble *get_Y(const MatrixDouble *mat)
{
    auto *Y = (MatrixDouble *) malloc(sizeof(MatrixDouble) * mat->rows);
    Y->rows = mat->rows;
    Y->cols = 1;
    cblas_dcopy(mat->rows, mat->data, mat->cols, Y->data, 1);
    return Y;
}

int main()
{
    CSimpleIniA ini;
    ini.SetUnicode();

    const SI_Error rc = ini.LoadFile("config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    }
    SI_ASSERT(rc == SI_OK);

    const auto num_generations = (int) ini.GetLongValue("MNIST", "GENERATIONS", 5);
    const auto max_items = (int) ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    const bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    const auto alpha = (float) ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    const auto hidden_layer_size = (int) ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    const std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    const std::string save_dir = base_dir + "/train";
    const std::string img_filename =
            ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    const std::string img_path = base_dir + "/" + img_filename;
    const std::string label_filename =
            ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    const std::string label_path = base_dir + "/" + label_filename;

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

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    auto train_mat = to_openblas(train_dataset._images);

    auto Y_train_float = get_Y(train_mat);

    free(Y_train_float);
    free(train_mat);
    return 0;
}
