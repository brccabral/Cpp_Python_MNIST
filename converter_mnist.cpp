#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <NeuralNet/NeuralNet.hpp>

#ifdef DEBUG
#define DUMP_VAR(x) std::cout << "---\n" \
                              << #x "\n" \
                              << x << "\n---" << std::endl

float get_float_matrix(Eigen::MatrixXf &M, int r, int c)
{
    return M(r, c);
}

float get_float_vector(Eigen::VectorXf &V, int r)
{
    return V(r);
}

#else
#define DUMP_VAR(x)
#endif

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

    NeuralNet neural_net(X_train.cols(), hidden_layer_size, categories);
    Eigen::MatrixXf one_hot_Y = NeuralNet::one_hot_encode(Y_train);

    Eigen::MatrixXf output;

    int correct_prediction = 0;
    float acc = 0.0f;

    Eigen::VectorXf prediction;

    for (int generation = 0; generation < num_generations; generation++)
    {
        output = neural_net.forward_prop(X_train_T);

        if (generation % 50 == 0)
        {
            prediction = NeuralNet::get_predictions(output);
            correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_train);
            acc = NeuralNet::get_accuracy(correct_prediction, Y_train.rows());
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }

        neural_net.back_prop(X_train_T, Y_train, one_hot_Y, alpha);
    }
    prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_train);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_train.rows());
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    save_dir = base_dir + "/test";
    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images.idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels.idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    MNIST_Dataset test_dataset(img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);

    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");

    Eigen::MatrixXf test_mat = test_dataset.to_matrix();

    Eigen::MatrixXf X_test = test_mat.leftCols(test_mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_test = test_mat.rightCols(1);                  // n,1
    X_test = X_test / 255.0;

    Eigen::MatrixXf X_test_T = X_test.transpose();

    output = neural_net.forward_prop(X_test_T);

    prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_test);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_test.rows());
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);
}