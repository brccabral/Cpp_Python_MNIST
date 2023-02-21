#include <SimpleIni/SimpleIni.h>
#include <torch/torch.h>
#include <iostream>

// Define a new Module.
struct Net : torch::nn::Module
{
    Net()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

// Define a new Module.
struct Net2 : torch::nn::Module
{
    Net2()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 10));
        fc3 = register_module("fc3", torch::nn::Linear(10, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main()
{
    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    };
    SI_ASSERT(rc == SI_OK);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    int batch_size = ini.GetLongValue("TORCH", "BATCH_SIZE", 64);
    int num_epochs = ini.GetLongValue("TORCH", "EPOCHS", 10);
    std::string save_model = ini.GetValue("TORCH", "SAVE_CPP", "net_cpp.pt");

    // Create a new Net.
    // auto net = std::make_shared<Net>();
    // torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
    auto net = std::make_shared<Net2>();
    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.1);
    torch::nn::NLLLoss loss_fn;

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST(base_dir).map(torch::data::transforms::Stack<>()),
        batch_size);

    std::tuple<torch::Tensor, torch::Tensor> tm;
    torch::Tensor prediction, values, indices, correct_bool;
    int correct_prediction;
    float acc;

    net->train();
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = loss_fn(prediction, batch.target); // (64x10, 64x1)
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (batch_index++ % 100 == 0)
            {
                // Serialize your model periodically as a checkpoint.
                torch::save(net, save_model);

                tm = torch::max(prediction, 1);
                std::tie(values, indices) = tm;

                correct_bool = batch.target == indices;
                correct_prediction = correct_bool.sum().item<int>();

                acc = 1.0f * correct_prediction / batch_size;
                printf("Epoch %lu\t Batch %lu\t Correct %d\tAccuracy %.4f\t Loss %.4f\n", epoch, batch_index, correct_prediction, acc, loss.item<float>());
            }
        }
    }

    auto net_loaded = std::make_shared<Net2>();
    torch::load(net_loaded, save_model);
    net_loaded->eval();

    auto test_dataset = torch::data::datasets::MNIST(base_dir, torch::data::datasets::MNIST::Mode::kTest);
    c10::optional<size_t> test_size = test_dataset.size();
    int size = int(test_size.value());
    std::cout << size << std::endl;

    // torch::Tensor x_tensor = torch::empty({size, 784});
    // torch::Tensor y_tensor = torch::empty(size);
    // for (int b = 0; b < size; b++)
    // {
    //     x_tensor.index_put_({b}, test_dataset.get(b).data.reshape({784}));
    //     y_tensor.index_put_({b}, test_dataset.get(b).target);
    // }
    // torch::Tensor y_tensor_i = y_tensor.toType(c10::ScalarType::Long);
    // x_tensor.set_requires_grad(true);

    // std::cout << x_tensor.sizes() << std::endl;
    // std::cout << y_tensor_i.sizes() << std::endl;

    // prediction = net->forward(x_tensor);
    // correct_bool = y_tensor_i == indices;

    auto data_tester = torch::data::make_data_loader(
        test_dataset.map(torch::data::transforms::Stack<>()),
        batch_size);

    correct_prediction = 0;
    for (auto &batch_test : *data_tester)
    {
        prediction = net_loaded->forward(batch_test.data);

        tm = torch::max(prediction, 1);
        std::tie(values, indices) = tm;

        correct_bool = batch_test.target == indices;
        correct_prediction += correct_bool.sum().item<int>();
    }
    acc = 1.0f * correct_prediction / size;
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}