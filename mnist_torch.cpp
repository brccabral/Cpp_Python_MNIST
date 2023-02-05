#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <torch/torch.h>

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

int main(int argc, char *argv[])
{
    std::srand((unsigned int)time(0));

    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("../../config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    };
    SI_ASSERT(rc == SI_OK);

    int num_generations = ini.GetLongValue("MNIST", "GENERATIONS", 5);
    int max_items = ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    float alpha = ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST");
    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images.idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename = ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels.idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

    MNIST_Dataset train_dataset(img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    Eigen::MatrixXf train_mat = train_dataset.to_matrix();

    Eigen::MatrixXf X_train = train_mat.leftCols(train_mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_train = train_mat.rightCols(1);                   // n,1
    X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    Eigen::MatrixXf X_train_T = X_train.transpose();

    torch::Tensor inputElement = torch::from_blob(X_train_T.data(), {X_train_T.rows(), X_train_T.cols()}).clone();

    return EXIT_SUCCESS;
}
