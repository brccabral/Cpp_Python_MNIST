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

int main()
{
    // Create a new Net.
    auto net = std::make_shared<Net>();

    int batch_size = 64;
    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("../../MNIST_data").map(torch::data::transforms::Stack<>()),
        /*batch_size=*/batch_size);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    net->train();
    for (size_t epoch = 1; epoch <= 10; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *data_loader)
        {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target); // (64x10, 64x1)
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (batch_index++ % 100 == 0)
            {
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");

                std::tuple<torch::Tensor, torch::Tensor> tm = torch::max(prediction, 1);
                torch::Tensor values, indices;
                std::tie(values, indices) = tm;

                torch::Tensor correct_bool = batch.target == indices;
                int correct_prediction = correct_bool.sum().item<int>();

                float acc = 1.0f * correct_prediction / batch_size;
                printf("Epoch %lu\t Batch %lu\t Correct %d\tAccuracy %.4f\t Loss %.4f\n", epoch, batch_index, correct_prediction, acc, loss.item<float>());
            }
        }
    }

    return EXIT_SUCCESS;
}