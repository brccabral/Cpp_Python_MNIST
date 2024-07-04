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
    size_t num_epochs = ini.GetLongValue("TORCH", "EPOCHS", 10);
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
        train_dataset.map(torch::data::transforms::Stack<>()),
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
    for (size_t epoch = 0; epoch < num_epochs; ++epoch)
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
                float loss_value = loss.item<float>();
                size_t current = batch_index * X.sizes()[0];

                tm = torch::max(prediction, 1);
                values = std::get<0>(tm);
                indices = std::get<1>(tm);

                correct_bool = y == indices;
                correct_prediction = correct_bool.sum().item<int>();

                acc = 1.0f * (float)correct_prediction / (float)batch_size;
                printf("Epoch: %lu \t Batch: %lu \t Correct: %d \t Accuracy: %.4f \t Loss: %.4f \t [%5lu/%5lu] \n", epoch, batch_index, correct_prediction, acc, loss_value, current, size);

                // Serialize your model periodically as a checkpoint.
                torch::save(net, save_model);
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

    int num_batches = 0;
    float test_loss = 0.0f;
    int correct = 0;
    acc = 0.0f;

    {
        torch::NoGradGuard no_grad;

        for (auto &batch_test : *test_dataloader)
        {
            torch::Tensor X = batch_test.data.to(device);
            torch::Tensor y = batch_test.target.to(device);
            X = flatten(X);

            prediction = net_loaded->forward(X);
            torch::Tensor loss = loss_fn(prediction, y);
            float loss_value = loss.item<float>();

            tm = torch::max(prediction, 1);
            values = std::get<0>(tm);
            indices = std::get<1>(tm);

            correct_bool = y == indices;
            correct_prediction = correct_bool.sum().item<int>();

            test_loss += loss_value;
            correct += correct_prediction;
            num_batches++;
        }
    }
    test_loss /= (float)num_batches;

    size = test_dataset.size().value();
    acc = 1.0f * (float)correct / (float)size;

    printf("Test: Correct: %d \t Accuracy: %.4f \t Loss: %.4f \t [%5lu] \n", correct, acc, test_loss, size);

    return EXIT_SUCCESS;
}