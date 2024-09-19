#include "NumCpp.hpp"
#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049


int main()
{
    std::srand((unsigned int)time(0));

    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    };
    SI_ASSERT(rc == SI_OK);

    int num_generations = (int)ini.GetLongValue("MNIST", "GENERATIONS", 5);
    int max_items = (int)ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    float alpha = (float)ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = (int)ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename = ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

    std::cout << PRINT_VAR(num_generations) << " "
              << PRINT_VAR(max_items) << " "
              << PRINT_VAR(save_img) << " "
              << PRINT_VAR(alpha) << " "
              << PRINT_VAR(hidden_layer_size) << " "
              << PRINT_VAR(base_dir) << " "
              << PRINT_VAR(save_dir) << " "
              << PRINT_VAR(img_filename) << " "
              << PRINT_VAR(img_path) << " "
              << PRINT_VAR(label_filename) << " "
              << PRINT_VAR(label_path) << " "
              << std::endl;

    MNIST_Dataset train_dataset(img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);
    std::cout << PRINT_VAR(train_dataset.get_images_length())
              << std::endl;
    std::cout << PRINT_VAR(train_dataset.get_label_from_index(4))
              << std::endl;

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    nc::NdArray<float> train_mat = train_dataset.to_numcpp();
    nc::NdArray<float> Y_train = MNIST_Dataset::get_Y(train_mat);
    nc::NdArray<float> X_train = MNIST_Dataset::get_X(train_mat);

    std::cout << Y_train(4, 0) << std::endl;
    std::cout << X_train.numRows() << "," << X_train.numCols() << std::endl;
    for (int c = 0; c < X_train.numCols(); c++)
        std::cout << X_train(4, c) << ", ";
    std::cout << std::endl;

    return 0;
}
