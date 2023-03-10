#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <torch/torch.h>
#include <TorchNet/torchnet.hpp>

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

// https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156/6
torch::Tensor eigenMatrixToTorchTensor(Eigen::MatrixXf e)
{
    auto t = torch::empty({e.cols(), e.rows()});
    float *data = t.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> ef(data, t.size(1), t.size(0));
    ef = e.cast<float>();
    return t.transpose(0, 1);
}

torch::Tensor eigenVectorToTorchTensor(Eigen::VectorXf e)
{
    auto t = torch::empty({e.rows()});
    float *data = t.data_ptr<float>();

    Eigen::Map<Eigen::VectorXf> ef(data, t.size(0), 1);
    ef = e.cast<float>();
    return t;
}

int main(int argc, char *argv[])
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

    int num_generations = ini.GetLongValue("MNIST", "GENERATIONS", 5);
    int max_items = ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    float alpha = ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");

    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename = ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

    std::cout << "Reading dataset file" << std::endl;
    MNIST_Dataset train_dataset(img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    std::cout << "Converting to matrix" << std::endl;
    Eigen::MatrixXf train_mat = train_dataset.to_matrix();

    Eigen::VectorXf Y_train = MNIST_Dataset::get_Y(train_mat);
    Eigen::MatrixXf X_train = MNIST_Dataset::get_X(train_mat);
    X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    std::cout << "Preparing tensors" << std::endl;

    torch::Tensor x_tensor_train = eigenMatrixToTorchTensor(X_train);
    torch::Tensor y_tensor_train = eigenVectorToTorchTensor(Y_train);

    std::cout << x_tensor_train.sizes() << std::endl;
    // std::cout << x_tensor.index({0, torch::indexing::Slice(0,783)}) << std::endl;
    std::cout << y_tensor_train.sizes() << std::endl;
    torch::Tensor y_tensor_i_train = y_tensor_train.toType(c10::ScalarType::Long);

    x_tensor_train.set_requires_grad(true);

    Net net = Net(X_train.cols(), hidden_layer_size, categories);
    net.train();

    torch::optim::SGD optimizer(net.parameters(), /*lr=*/alpha);
    torch::nn::NLLLoss loss_fn;

    torch::Tensor values_train, indices_train, prediction_train, correct_bool_train;
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
            std::tie(values_train, indices_train) = tm_train;

            correct_bool_train = y_tensor_i_train == indices_train;
            correct_prediction = correct_bool_train.sum().item<int>();

            acc = 1.0f * correct_prediction / Y_train.rows();
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }
    }
    prediction_train = net.forward(x_tensor_train);

    tm_train = torch::max(prediction_train, 1);
    std::tie(values_train, indices_train) = tm_train;

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
    MNIST_Dataset test_dataset(img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);

    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");

    Eigen::MatrixXf test_mat = test_dataset.to_matrix();

    Eigen::VectorXf Y_test = MNIST_Dataset::get_Y(test_mat);
    Eigen::MatrixXf X_test = MNIST_Dataset::get_X(test_mat);
    X_test = X_test / 255.0;

    torch::Tensor x_tensor_test = eigenMatrixToTorchTensor(X_test);
    torch::Tensor y_tensor_test = eigenVectorToTorchTensor(Y_test);
    torch::Tensor y_tensor_i_test = y_tensor_test.toType(c10::ScalarType::Long);

    torch::Tensor values_test, indices_test, prediction_test, correct_bool_test;
    std::tuple<torch::Tensor, torch::Tensor> tm_test;

    prediction_test = net.forward(x_tensor_test);

    tm_test = torch::max(prediction_test, 1);
    std::tie(values_test, indices_test) = tm_test;

    correct_bool_test = y_tensor_i_test == indices_test;
    correct_prediction = correct_bool_test.sum().item<int>();

    acc = 1.0f * correct_prediction / Y_test.rows();
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}
