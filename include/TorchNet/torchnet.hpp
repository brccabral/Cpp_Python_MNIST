#pragma once

#include <torch/torch.h>

// https://pytorch.org/cppdocs/frontend.html#end-to-end-example
struct NetImpl : torch::nn::Module
{
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};

    // create one hidden layer having hidden_layer_size,
    // out has categories size
    NetImpl(int num_features,
            int hidden_layer_size,
            int categories)
    {
        fc1 = register_module("fc1", torch::nn::Linear(num_features, hidden_layer_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_layer_size, hidden_layer_size / 2));
        out = register_module("out", torch::nn::Linear(hidden_layer_size / 2, categories));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(out->forward(x), 1);
        return x;
    }
};

TORCH_MODULE(Net);