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
    float db1;
    nc::NdArray<float> dW2;
    float db2;

public:
    NeuralNetNC(unsigned int num_features,
              unsigned int hidden_layer_size,
              unsigned int categories);

    static nc::NdArray<float> ReLU(nc::NdArray<float> &Z);

    static nc::NdArray<float> Softmax(nc::NdArray<float> &Z);

    nc::NdArray<float> forward_prop(nc::NdArray<float> &X);

    static nc::NdArray<int> one_hot_encode(nc::NdArray<int> &Y);

    static nc::NdArray<bool> deriv_ReLU(nc::NdArray<float> &Z);

    void back_prop(
        nc::NdArray<float> &X,
        nc::NdArray<float> &Y,
        nc::NdArray<float> &one_hot_Y,
        float alpha);

    static nc::NdArray<unsigned> get_predictions(nc::NdArray<float> &P);

    static int get_correct_prediction(nc::NdArray<unsigned> &p, nc::NdArray<int> &y);

    static float get_accuracy(int correct_prediction, int size);
};
