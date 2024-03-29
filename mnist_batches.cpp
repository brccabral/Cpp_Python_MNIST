#include <SimpleIni/SimpleIni.h>
#include <torch/torch.h>
#include <iostream>
#include <TorchNet/torchnet.hpp>

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
    float alpha = ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    // Create a multi-threaded data loader for the MNIST dataset.
    auto train_dataset = torch::data::datasets::MNIST(base_dir);
    std::cout << "train_dataset.images().sizes()=" << train_dataset.images().sizes() << std::endl;
    size_t size = train_dataset.size().value();
    std::cout << "train size=" << size << std::endl;

    int categories = train_dataset.targets().max().item<int>() + 1;
    int num_features = torch::tensor(train_dataset.images().sizes().slice(1)).prod().item<int>();

    std::cout << "num_features=" << num_features << std::endl;
    std::cout << "categories=" << categories << std::endl;

    auto train_dataloader = torch::data::make_data_loader(
        torch::data::datasets::MNIST(base_dir).map(torch::data::transforms::Stack<>()),
        batch_size);

    // Create a new Net.
    // auto net = std::make_shared<Net>();
    // torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
    auto net = std::make_shared<Net>(num_features, hidden_layer_size, categories);
    std::cout << *net << std::endl;

    c10::DeviceType device;

    if (torch::cuda::is_available())
    {
        std::cout << "Using cuda" << std::endl;
        // Move model and inputs to GPU
        device = torch::kCUDA;
    }
    else
    {
        device = torch::kCPU;
    }
    net->to(device);

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/alpha);
    torch::nn::NLLLoss loss_fn;
    torch::nn::Flatten flatten;

    std::tuple<torch::Tensor, torch::Tensor> tm;
    torch::Tensor prediction, values, indices, correct_bool;
    int correct_prediction;
    float acc;

    net->train();
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch)
    {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto &batch : *train_dataloader)
        {
            torch::Tensor X = batch.data.to(device);
            torch::Tensor y = batch.target.to(device);
            X = flatten(X);

            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            prediction = net->forward(X);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = loss_fn(prediction, y); // (64x10, 64x1)
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

                correct_bool = y == indices;
                correct_prediction = correct_bool.sum().item<int>();

                acc = 1.0f * correct_prediction / batch_size;
                printf("Epoch %lu\t Batch %lu\t Correct %d\tAccuracy %.4f\t Loss %.4f\n", epoch, batch_index, correct_prediction, acc, loss.item<float>());
            }
        }
    }
    torch::save(net, save_model);

    auto net_loaded = std::make_shared<Net>(num_features, hidden_layer_size, categories);
    torch::load(net_loaded, save_model);
    net_loaded->eval();

    auto test_dataset = torch::data::datasets::MNIST(base_dir, torch::data::datasets::MNIST::Mode::kTest);
    size = test_dataset.size().value();
    std::cout << size << std::endl;
    std::cout << test_dataset.images().data().sizes() << std::endl;

    auto test_dataloader = torch::data::make_data_loader(
        test_dataset.map(torch::data::transforms::Stack<>()),
        batch_size);

    correct_prediction = 0;
    for (auto &batch_test : *test_dataloader)
    {
        torch::Tensor X = batch_test.data.to(device);
        torch::Tensor y = batch_test.target.to(device);
        X = flatten(X);

        prediction = net_loaded->forward(X);

        tm = torch::max(prediction, 1);
        std::tie(values, indices) = tm;

        correct_bool = y == indices;
        correct_prediction += correct_bool.sum().item<int>();
    }
    acc = 1.0f * correct_prediction / size;
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    return EXIT_SUCCESS;
}