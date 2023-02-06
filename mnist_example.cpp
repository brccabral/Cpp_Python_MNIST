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
    auto dataset = torch::data::datasets::MNIST("../../MNIST_data");
    c10::optional<size_t> size = dataset.size();
    std::cout << int(size.value()) << std::endl;

    // std::cout << dataset.get(0).data << std::endl; // (1 x 28 x 28)

    // auto batch2 = dataset.get_batch({0, torch::indexing::None});
    // std::cout << batch2.at(0).data.sizes() << std::endl;
    // std::cout << batch2.at(0).data << std::endl;
    // std::cout << batch2.at(0).target << std::endl;
    torch::Tensor x_tensor = torch::zeros({60000, 784});
    torch::Tensor y_tensor = torch::zeros(60000);
    for(int b = 0 ; b< 60000; b++){
        // std::cout << b << std::endl;
        // auto batch2 = dataset.get(b);
        // auto d = batch2.data;
        // std::cout << d.sizes() << std::endl;
        x_tensor.index_put_({b}, dataset.get(b).data.reshape({784}));
        y_tensor.index_put_({b}, dataset.get(b).target);
    }
    torch::Tensor y_tensor_i = y_tensor.toType(c10::ScalarType::Long);

    std::cout << x_tensor.sizes() << std::endl;
    std::cout << y_tensor_i.sizes() << std::endl;
    // std::vector<torch::data::Example<at::Tensor, at::Tensor>
    // torch::Tensor batch_tensor = torch::from_blob(batch2.data(),
    //                                           {60000, 784},
    //                                           torch::requires_grad())
    //                              .clone();
    // std::cout << batch_tensor.size(0) << std::endl;
    // std::cout << batch_tensor.size(1) << std::endl;
    // return 0;

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
        torch::Tensor prediction = net2.forward(x_tensor);
        torch::Tensor loss = torch::nll_loss(prediction, y_tensor_i);
        loss.backward();
        optimizer.step();
        if (generation % 50 == 0)
        {
            tm = torch::max(prediction, 1);
            std::tie(values, indices) = tm;

            correct_bool = y_tensor_i == indices;
            correct_prediction = correct_bool.sum().item<int>();

            acc = 1.0f * correct_prediction / 60000;
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }
    }

    // Create a new Net.
    // auto net = std::make_shared<Net>();

    // int batch_size = 512;
    // // Create a multi-threaded data loader for the MNIST dataset.
    // auto data_loader = torch::data::make_data_loader(
    //     torch::data::datasets::MNIST("../../MNIST_data").map(torch::data::transforms::Stack<>()),
    //     // /*batch_size=*/64);
    //     /*batch_size=*/batch_size);

    // // Instantiate an SGD optimization algorithm to update our Net's parameters.
    // torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    // net->train();
    // for (size_t epoch = 1; epoch <= 10; ++epoch)
    // {
    //     size_t batch_index = 0;
    //     // Iterate the data loader to yield batches from the dataset.
    //     for (auto &batch : *data_loader)
    //     {
    //         // Reset gradients.
    //         optimizer.zero_grad();
    //         // Execute the model on the input data.
    //         torch::Tensor prediction = net->forward(batch.data);
    //         // Compute a loss value to judge the prediction of our model.
    //         torch::Tensor loss = torch::nll_loss(prediction, batch.target); // (64x10, 64x1)
    //         // Compute gradients of the loss w.r.t. the parameters of our model.
    //         loss.backward();
    //         // Update the parameters based on the calculated gradients.
    //         optimizer.step();
    //         // Output the loss and checkpoint every 100 batches.
    //         if (batch_index++ % 100 == 0)
    //         {
    //             // Serialize your model periodically as a checkpoint.
    //             torch::save(net, "net.pt");

    //             std::tuple<torch::Tensor, torch::Tensor> tm = torch::max(prediction, 1);
    //             torch::Tensor values, indices;
    //             std::tie(values, indices) = tm;

    //             torch::Tensor correct_bool = batch.target == indices;
    //             int correct_prediction = correct_bool.sum().item<int>();

    //             float acc = 1.0f * correct_prediction / batch_size;
    //             printf("Epoch %lu\t Batch %lu\t Correct %d\tAccuracy %.4f\t Loss %.4f\n", epoch, batch_index, correct_prediction, acc, loss.item<float>());
    //             break;
    //         }
    //     }
    // }

    // return 0;
}