#pragma once

#include "NumCpp.hpp"


class NeuralNetNC
{
private:

    // layers
    nc::NdArray<float> W1;
    nc::NdArray<float> b1;
    nc::NdArray<float> W2;
    nc::NdArray<float> b2;

    // back prop
    nc::NdArray<float> Z1;
    nc::NdArray<float> A1;
    nc::NdArray<float> Z2;
    nc::NdArray<float> A2;

    // gradients
    nc::NdArray<float> dW1;
    float db1{};
    nc::NdArray<float> dW2;
    float db2{};

public:

    NeuralNetNC(unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories);

    static nc::NdArray<float> ReLU(const nc::NdArray<float> &Z);

    static nc::NdArray<float> Softmax(const nc::NdArray<float> &Z);

    nc::NdArray<float> forward_prop(const nc::NdArray<float> &X);

    static nc::NdArray<int> one_hot_encode(nc::NdArray<int> &Y);

    static nc::NdArray<float> deriv_ReLU(const nc::NdArray<float> &Z);

    void back_prop(
            const nc::NdArray<float> &X, const nc::NdArray<int> &Y,
            const nc::NdArray<int> &one_hot_Y, float alpha);

    static nc::NdArray<unsigned> get_predictions(const nc::NdArray<float> &P);

    static int get_correct_prediction(const nc::NdArray<unsigned> &p, const nc::NdArray<int> &y);

    static float get_accuracy(int correct_prediction, int size);

    static void rnd_seed(int seed);
};
