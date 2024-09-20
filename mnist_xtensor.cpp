#include <iostream>
#include <MNIST/MNIST_Dataset.hpp>
#include <SimpleIni/SimpleIni.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049


int main()
{
    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("config.ini");
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

    auto train_mat = train_dataset.to_xtensor();

    xt::xarray<float> Y_train_float = MNIST_Dataset::get_Y(train_mat);
    xt::xarray<float> X_train = MNIST_Dataset::get_X(train_mat);
    X_train /= 255.0f;
    std::cout << Y_train_float(4) << std::endl;
    std::cout << X_train.shape()[0] << "," << X_train.shape()[1] << std::endl;
    std::cout << xt::view(X_train, 4, xt::all()) << std::endl;

    return 0;
}
