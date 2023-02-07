#include <torch/torch.h>
#include <iostream>

// Define a new Module.
struct Net : torch::nn::Module
{
    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, out{nullptr};

    Net()
    {
        // Construct and register two Linear submodules.
        // fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        // fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        // fc3 = register_module("fc3", torch::nn::Linear(32, 10));
        fc1 = register_module("fc1", torch::nn::Linear(784, 10));
        out = register_module("fc3", torch::nn::Linear(10, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        // x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        // x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        // x = torch::relu(fc2->forward(x));
        // x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::log_softmax(out->forward(x), /*dim=*/1);
        return x;
    }
};

int main()
{
    auto dataset = torch::data::datasets::MNIST("../../MNIST_data/MNIST/raw");
    c10::optional<size_t> train_size = dataset.size();
    int size = int(train_size.value());
    std::cout << size << std::endl;

    torch::Tensor x_tensor = torch::empty({size, 784});
    torch::Tensor y_tensor = torch::empty(size);
    for(int b = 0 ; b < size; b++){
        x_tensor.index_put_({b}, dataset.get(b).data.reshape({784}));
        y_tensor.index_put_({b}, dataset.get(b).target);
    }
    torch::Tensor y_tensor_i = y_tensor.toType(c10::ScalarType::Long);
    x_tensor.set_requires_grad(true);

    std::cout << x_tensor.sizes() << std::endl;
    std::cout << y_tensor_i.sizes() << std::endl;

    Net net2 = Net();
    net2.train();

    torch::optim::SGD optimizer(net2.parameters(), /*lr=*/0.01);
    std::tuple<torch::Tensor, torch::Tensor> tm;
    torch::Tensor values, indices, prediction, correct_bool;
    int correct_prediction = 0;
    float acc = 0.0f;

    for (int generation = 0; generation < 500; generation++)
    {
        optimizer.zero_grad();
        prediction = net2.forward(x_tensor);
        torch::Tensor loss = torch::nll_loss(prediction, y_tensor_i);
        loss.backward();
        optimizer.step();
        if (generation % 50 == 0)
        {
            tm = torch::max(prediction, 1);
            std::tie(values, indices) = tm;

            correct_bool = y_tensor_i == indices;
            correct_prediction = correct_bool.sum().item<int>();

            acc = 1.0f * correct_prediction / size;
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }
    }
    prediction = net2.forward(x_tensor);

    tm = torch::max(prediction, 1);
    std::tie(values, indices) = tm;

    correct_bool = y_tensor_i == indices;
    correct_prediction = correct_bool.sum().item<int>();

    acc = 1.0f * correct_prediction / size;
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);


    auto test_dataset = torch::data::datasets::MNIST("../../MNIST_data/MNIST/raw", torch::data::datasets::MNIST::Mode::kTest);
    c10::optional<size_t> test_size = test_dataset.size();
    size = int(test_size.value());
    std::cout << size << std::endl;

    x_tensor = torch::empty({size, 784});
    y_tensor = torch::empty(size);
    for(int b = 0 ; b < size; b++){
        x_tensor.index_put_({b}, test_dataset.get(b).data.reshape({784}));
        y_tensor.index_put_({b}, test_dataset.get(b).target);
    }
    y_tensor_i = y_tensor.toType(c10::ScalarType::Long);
    x_tensor.set_requires_grad(true);

    std::cout << x_tensor.sizes() << std::endl;
    std::cout << y_tensor_i.sizes() << std::endl;

    net2.eval();
    prediction = net2.forward(x_tensor);

    tm = torch::max(prediction, 1);
    std::tie(values, indices) = tm;

    correct_bool = y_tensor_i == indices;
    correct_prediction = correct_bool.sum().item<int>();

    acc = 1.0f * correct_prediction / size;
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}