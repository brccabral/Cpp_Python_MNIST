#if _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <SimpleIni/SimpleIni.h>
#include <iostream>
#include <MNIST/MNIST_Dataset.hpp>
#include <Eigen/Dense>
#include <torch/torch.h>
#include <TorchNet/torchnet.hpp>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

// https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156/6
torch::Tensor eigenMatrixToTorchTensor(const Eigen::MatrixXf &e)
{
    const auto t = torch::empty({e.cols(), e.rows()});
    float *data = t.data_ptr<float>(); // NOLINT(*-use-auto)

    // the `data` is a pointer, and Eigen::Map<> populates `t` in-place
    // ReSharper disable once CppDFAUnusedValue
    // ReSharper disable once CppEntityAssignedButNoRead
    // ReSharper disable once CppDFAUnreadVariable
    Eigen::Map<Eigen::MatrixXf> ef(data, t.size(1), t.size(0));
    // ReSharper disable once CppDFAUnusedValue
    ef = e.cast<float>();
    return t.transpose(0, 1);
}

torch::Tensor eigenVectorToTorchTensor(const Eigen::VectorXf &e)
{
    auto t = torch::empty({e.rows()});
    float *data = t.data_ptr<float>(); // NOLINT(*-use-auto)

    // the `data` is a pointer, and Eigen::Map<> populates `t` in-place
    // ReSharper disable once CppDFAUnusedValue
    // ReSharper disable once CppEntityAssignedButNoRead
    // ReSharper disable once CppDFAUnreadVariable
    Eigen::Map<Eigen::VectorXf> ef(data, t.size(0), 1);
    // ReSharper disable once CppDFAUnusedValue
    ef = e.cast<float>();
    return t;
}

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

    int num_generations = ini.GetLongValue("MNIST", "GENERATIONS", 5);
    int max_items = ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    float alpha = ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

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

    std::cout << "Reading dataset file" << std::endl;
    MNIST_Dataset train_dataset(
            img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);
    std::cout << PRINT_VAR(train_dataset.get_images_length()) << std::endl;
    std::cout << PRINT_VAR(train_dataset.get_label_from_index(4)) << std::endl;

#ifdef CV_SAVE_IMAGES
    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);
    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");
#endif

    std::cout << "Converting to matrix" << std::endl;
    Eigen::MatrixXf train_mat = to_matrix(train_dataset._images);

    Eigen::VectorXf Y_train = get_Y(train_mat);
    Eigen::MatrixXf X_train = get_X(train_mat);
    std::cout << Y_train(4) << std::endl;
    std::cout << X_train.rows() << "," << X_train.cols() << std::endl;
    for (int c = 0; c < X_train.cols(); c++)
        std::cout << X_train(4, c) << ", ";
    std::cout << std::endl;

    X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    std::cout << "Preparing tensors" << std::endl;

    torch::Tensor x_tensor_train = eigenMatrixToTorchTensor(X_train);
    torch::Tensor y_tensor_train = eigenVectorToTorchTensor(Y_train);

    std::cout << x_tensor_train.sizes() << std::endl;
    // std::cout << x_tensor.index({0, torch::indexing::Slice(0,783)}) << std::endl;
    std::cout << y_tensor_train.sizes() << std::endl;
    torch::Tensor y_tensor_i_train = y_tensor_train.toType(c10::ScalarType::Long);

    x_tensor_train = x_tensor_train.set_requires_grad(true);

    auto net = Net(X_train.cols(), hidden_layer_size, categories);
    net.train();

    torch::optim::SGD optimizer(net.parameters(), /*lr=*/alpha);
    torch::nn::NLLLoss loss_fn;

    torch::Tensor indices_train, prediction_train, correct_bool_train;
    std::tuple<torch::Tensor, torch::Tensor> tm_train;
    int correct_prediction = 0;
    float acc = 0.0f;

    for (int generation = 0; generation < num_generations; generation++)
    {
        optimizer.zero_grad();
        prediction_train = net.forward(x_tensor_train);
        torch::Tensor loss = loss_fn(prediction_train, y_tensor_i_train);
        loss.backward();
        optimizer.step();

        if (generation % 50 == 0)
        {
            tm_train = torch::max(prediction_train, 1);
            indices_train = std::get<1>(tm_train);

            correct_bool_train = y_tensor_i_train == indices_train;
            correct_prediction = correct_bool_train.sum().item<int>();

            acc = 1.0f * (float) correct_prediction / (float) Y_train.rows();
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }
    }
    prediction_train = net.forward(x_tensor_train);

    tm_train = torch::max(prediction_train, 1);
    indices_train = std::get<1>(tm_train);

    correct_bool_train = y_tensor_train == indices_train;
    correct_prediction = correct_bool_train.sum().item<int>();

    acc = 1.0f * correct_prediction / Y_train.rows();
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    save_dir = base_dir + "/test";
    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    net.eval();
    MNIST_Dataset test_dataset(
            img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

#ifdef CV_SAVE_IMAGES
    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);
    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");
#endif

    Eigen::MatrixXf test_mat = to_matrix(test_dataset._images);

    Eigen::VectorXf Y_test = get_Y(test_mat);
    Eigen::MatrixXf X_test = get_X(test_mat);
    X_test = X_test / 255.0;

    torch::Tensor x_tensor_test = eigenMatrixToTorchTensor(X_test);
    torch::Tensor y_tensor_test = eigenVectorToTorchTensor(Y_test);
    torch::Tensor y_tensor_i_test = y_tensor_test.toType(c10::ScalarType::Long);

    std::cout << x_tensor_test.sizes() << std::endl;
    std::cout << y_tensor_test.sizes() << std::endl;

    torch::Tensor indices_test, prediction_test, correct_bool_test;
    std::tuple<torch::Tensor, torch::Tensor> tm_test;

    {
        torch::NoGradGuard no_grad;
        prediction_test = net.forward(x_tensor_test);
    }

    tm_test = torch::max(prediction_test, 1);
    indices_test = std::get<1>(tm_test);

    correct_bool_test = y_tensor_i_test == indices_test;
    correct_prediction = correct_bool_test.sum().item<int>();

    acc = 1.0f * (float) correct_prediction / (float) Y_test.rows();
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}
