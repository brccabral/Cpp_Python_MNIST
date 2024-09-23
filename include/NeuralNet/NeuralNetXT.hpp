#pragma once

#include <xtensor/xarray.hpp>

class NeuralNetXT
{
private:

    // layers
    xt::xarray<float> W1;
    xt::xarray<float> b1;
    xt::xarray<float> W2;
    xt::xarray<float> b2;

    // back prop
    xt::xarray<float> Z1;
    xt::xarray<float> A1;
    xt::xarray<float> Z2;
    xt::xarray<float> A2;

public:

    NeuralNetXT(unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories);

    static xt::xarray<float> ReLU(const xt::xarray<float> &Z);

    static xt::xarray<float> Softmax(const xt::xarray<float> &Z);

    xt::xarray<float> forward_prop(const xt::xarray<float> &X);

    static xt::xarray<int> one_hot_encode(xt::xarray<int> &Y);

    static void rnd_seed(int seed);
};
