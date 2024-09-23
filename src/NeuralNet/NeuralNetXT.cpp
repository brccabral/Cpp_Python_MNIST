#include "NeuralNet/NeuralNetXT.hpp"
#include <xtensor/xrandom.hpp>


NeuralNetXT::NeuralNetXT(
        unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories)
{
    W1 = xt::random::rand<float>({hidden_layer_size, num_features}) - 0.5f;
    b1 = xt::random::rand<float>({hidden_layer_size, (unsigned int) 1}) - 0.5f;
    W2 = xt::random::rand<float>({categories, hidden_layer_size}) - 0.5f;
    b2 = xt::random::rand<float>({categories, (unsigned int) 1}) - 0.5f;
}

void NeuralNetXT::rnd_seed(const int seed)
{
    xt::random::seed(seed); // NOLINT(*-msc51-cpp)
}
