#if _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <iostream>
#include <MNIST/MNIST_Dataset.hpp>
#include <SimpleIni/SimpleIni.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <NeuralNetXT/NeuralNetXT.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049


xt::xarray<float> to_xtensor(const std::vector<MNIST_Image> &_images)
{
    const size_t number_images = _images.size();
    const size_t number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    const std::vector<size_t> shape = {number_images, number_pixels + 1};
    xt::xarray<float> mat(shape);
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

xt::xarray<float> get_X(const xt::xarray<float> &mat)
{
    using namespace xt::placeholders; // to enable _ syntax
    return xt::view(mat, xt::all(), xt::range(1, _));
}

xt::xarray<float> get_Y(const xt::xarray<float> &mat)
{
    using namespace xt::placeholders; // to enable _ syntax
    return xt::view(mat, xt::all(), 0);
}

int main()
{
    NeuralNetXT::rnd_seed((int) time(nullptr)); // NOLINT(*-msc51-cpp)

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


    auto train_mat = to_xtensor(train_dataset._images);

    xt::xarray<float> Y_train_float = get_Y(train_mat);
    xt::xarray<float> X_train = get_X(train_mat);
    X_train /= 255.0f;
    std::cout << Y_train_float(4) << std::endl;
    std::cout << X_train.shape()[0] << "," << X_train.shape()[1] << std::endl;
    std::cout << xt::view(X_train, 4, xt::all()) << std::endl;

    xt::xarray<int> Y_train = xt::cast<int>(Y_train_float);

    int categories = xt::amax(Y_train)() + 1;

    auto X_train_T = xt::transpose(X_train);

    auto neural_net = NeuralNetXT(X_train.shape()[1], hidden_layer_size, categories);
    auto one_hot_Y = NeuralNetXT::one_hot_encode(Y_train);

    xt::xarray<float> output;

    int correct_prediction = 0;
    float acc = 0.0f;

    xt::xarray<unsigned> prediction;

    for (int generation = 0; generation < num_generations; generation++)
    {
        output = neural_net.forward_prop(X_train_T);

        if (generation % 50 == 0)
        {
            prediction = NeuralNetXT::get_predictions(output);
            correct_prediction = NeuralNetXT::get_correct_prediction(prediction, Y_train);
            acc = NeuralNetXT::get_accuracy(correct_prediction, Y_train.size());
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }

        neural_net.back_prop(X_train, one_hot_Y, alpha);
    }


    output = neural_net.forward_prop(X_train_T);
    prediction = NeuralNetXT::get_predictions(output);
    correct_prediction = NeuralNetXT::get_correct_prediction(prediction, Y_train);
    acc = NeuralNetXT::get_accuracy(correct_prediction, Y_train.size());
    printf("Final\tCorrect %d\tAccuracy %.4f\n", correct_prediction, acc);

    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    MNIST_Dataset test_dataset(
            img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

#ifdef CV_SAVE_IMAGES
    save_dir = base_dir + "/test";
    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);
    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");
#endif // CV_SAVE_IMAGES


    auto test_mat = to_xtensor(test_dataset._images);

    auto Y_test_float = get_Y(test_mat);
    auto X_test = get_X(test_mat);
    X_test /= 255.0f;
    auto Y_test = xt::cast<int>(Y_test_float);

    auto X_test_T = xt::transpose(X_test);

    output = neural_net.forward_prop(X_test_T);

    prediction = NeuralNetXT::get_predictions(output);
    correct_prediction = NeuralNetXT::get_correct_prediction(prediction, Y_test);
    acc = NeuralNetXT::get_accuracy(correct_prediction, Y_test.size());
    printf("Test: \tCorrect %d\tAccuracy %.4f\n", correct_prediction, acc);

    return 0;
}
