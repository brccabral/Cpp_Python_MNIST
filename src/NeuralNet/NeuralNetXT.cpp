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

xt::xarray<int> NeuralNetXT::one_hot_encode(xt::xarray<int> &Y)
{
    xt::xarray<int> one_hot_Y = xt::zeros<int>({(int) Y.shape()[0], xt::amax(Y)() + 1});
    for (int i = 0; i < Y.shape()[0]; i++)
    {
        one_hot_Y(i, Y(i)) = 1;
    }
    return xt::transpose(one_hot_Y);
}

void NeuralNetXT::rnd_seed(const int seed)
{
    xt::random::seed(seed); // NOLINT(*-msc51-cpp)
}
