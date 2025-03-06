#include <SimpleIni/SimpleIni.h>
#include <iostream>
#include <MNIST/MNIST_Dataset.hpp>
#include <Eigen/Dense>
#include <NeuralNet/NeuralNet.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

Eigen::MatrixXf to_matrix(const std::vector<MNIST_Image> &_images)
{
    const int number_images = _images.size();
    const int number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    Eigen::MatrixXf mat(number_images, number_pixels + 1);
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

Eigen::MatrixXf get_X(Eigen::MatrixXf &mat)
{
    return mat.rightCols(mat.cols() - 1);
}

Eigen::VectorXf get_Y(Eigen::MatrixXf &mat)
{
    return mat.leftCols(1);
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


    Eigen::MatrixXf train_mat = to_matrix(train_dataset._images);

    Eigen::VectorXf Y_train = get_Y(train_mat);
    Eigen::MatrixXf X_train = get_X(train_mat);
    std::cout << Y_train(4) << std::endl;
    std::cout << X_train.rows() << "," << X_train.cols() << std::endl;
    for (int c = 0; c < X_train.cols(); c++)
        std::cout << X_train(4, c) << ", ";
    std::cout << std::endl;

    int categories = Y_train.maxCoeff() + 1;

    X_train = X_train / 255.0;
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
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }

        neural_net.back_prop(X_train, Y_train, one_hot_Y, alpha);
    }
    output = neural_net.forward_prop(X_train_T);
    prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_train);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_train.rows());
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    save_dir = base_dir + "/test";
    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    MNIST_Dataset test_dataset(
            img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

#ifdef CV_SAVE_IMAGES
    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);
    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");
#endif // CV_SAVE_IMAGES


    Eigen::MatrixXf test_mat = to_matrix(test_dataset._images);

    Eigen::VectorXf Y_test = get_Y(test_mat);
    Eigen::MatrixXf X_test = get_X(test_mat);
    X_test = X_test / 255.0;

    Eigen::MatrixXf X_test_T = X_test.transpose();

    output = neural_net.forward_prop(X_test_T);

    prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_test);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_test.rows());
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}
