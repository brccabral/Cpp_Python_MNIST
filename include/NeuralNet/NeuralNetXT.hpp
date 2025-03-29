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

    // gradients
    xt::xarray<float> dW1;
    float db1{};
    xt::xarray<float> dW2;
    float db2{};

public:

    NeuralNetXT(unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories);

    static xt::xarray<float> ReLU(const xt::xarray<float> &Z);

    static xt::xarray<float> Softmax(const xt::xarray<float> &Z);

    xt::xarray<float> forward_prop(const xt::xarray<float> &X);

    static xt::xarray<int> one_hot_encode(xt::xarray<int> &Y);

    static xt::xarray<float> deriv_ReLU(const xt::xarray<float> &Z);

    void back_prop(const xt::xarray<float> &X, const xt::xarray<int> &one_hot_Y, float alpha);

    static xt::xarray<int> get_predictions(const xt::xarray<float> &P);

    static int get_correct_prediction(const xt::xarray<int> &p, const xt::xarray<int> &y);

    static float get_accuracy(int correct_prediction, int size);

    static void rnd_seed(int seed);
};
