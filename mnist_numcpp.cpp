#if _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "NumCpp.hpp"
#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <NeuralNet/NeuralNetNC.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049


nc::NdArray<float> to_numcpp(const std::vector<MNIST_Image> &_images)
{
    const int number_images = _images.size();
    const int number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    nc::NdArray<float> mat(number_images, number_pixels + 1);
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

nc::NdArray<float> get_X(const nc::NdArray<float> &mat)
{
    int cols = mat.numCols();
    return mat(mat.rSlice(), {1, cols});
}

nc::NdArray<float> get_Y(const nc::NdArray<float> &mat)
{
    return mat(mat.rSlice(), {0});
}

int main()
{
    NeuralNetNC::rnd_seed((int) time(nullptr)); // NOLINT(*-msc51-cpp)

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


    nc::NdArray<float> train_mat = to_numcpp(train_dataset._images);

    nc::NdArray<float> Y_train_float = get_Y(train_mat);
    nc::NdArray<float> X_train = get_X(train_mat);
    X_train /= 255.0f;
    std::cout << Y_train_float(4, 0) << std::endl;
    std::cout << X_train.shape() << std::endl;
    std::cout << X_train(4, X_train.cSlice()) << std::endl;

    nc::NdArray<int> Y_train = Y_train_float.astype<int>();

    int categories = nc::max(Y_train).item() + 1;

    nc::NdArray<float> X_train_T = X_train.transpose();

    auto neural_net = NeuralNetNC(X_train.numCols(), hidden_layer_size, categories);
    nc::NdArray<int> one_hot_Y = NeuralNetNC::one_hot_encode(Y_train);

    nc::NdArray<float> output;

    int correct_prediction = 0;
    float acc = 0.0f;

    nc::NdArray<unsigned> prediction;

    for (int generation = 0; generation < num_generations; generation++)
    {
        output = neural_net.forward_prop(X_train_T);

        if (generation % 50 == 0)
        {
            prediction = NeuralNetNC::get_predictions(output);
            correct_prediction = NeuralNetNC::get_correct_prediction(prediction, Y_train);
            acc = NeuralNetNC::get_accuracy(correct_prediction, Y_train.size());
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }

        neural_net.back_prop(X_train, one_hot_Y, alpha);
    }

    output = neural_net.forward_prop(X_train_T);
    prediction = NeuralNetNC::get_predictions(output);
    correct_prediction = NeuralNetNC::get_correct_prediction(prediction, Y_train);
    acc = NeuralNetNC::get_accuracy(correct_prediction, Y_train.size());
    printf("Final\tCorrect %d\tAccuracy %.4f\n", correct_prediction, acc);

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


    auto test_mat = to_numcpp(test_dataset._images);

    auto Y_test_float = get_Y(test_mat);
    auto X_test = get_X(test_mat);
    X_test /= 255.0f;
    auto Y_test = Y_test_float.astype<int>();

    auto X_test_T = X_test.transpose();

    output = neural_net.forward_prop(X_test_T);

    prediction = NeuralNetNC::get_predictions(output);
    correct_prediction = NeuralNetNC::get_correct_prediction(prediction, Y_test);
    acc = NeuralNetNC::get_accuracy(correct_prediction, Y_test.size());
    printf("Test: \tCorrect %d\tAccuracy %.4f\n", correct_prediction, acc);

    return 0;
}
