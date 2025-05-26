#pragma once

#include <blaze/Blaze.h>

class NeuralNet_Blaze
{
public:

    NeuralNet_Blaze(int num_features, int hidden_layer_size, int categories);

    blaze::DynamicMatrix<double> forward_prop(const blaze::DynamicMatrix<double> &X);

    static blaze::DynamicMatrix<double> one_hot_encode(const blaze::DynamicMatrix<double> &Z);
    static blaze::DynamicMatrix<double> ReLU(const blaze::DynamicMatrix<double> &Z);

    blaze::DynamicMatrix<double> W1;
    blaze::DynamicVector<double> b1;
    blaze::DynamicMatrix<double> W2;
    blaze::DynamicVector<double> b2;
    blaze::DynamicMatrix<double> Z1;
    blaze::DynamicMatrix<double> A1;
    blaze::DynamicMatrix<double> Z2;
    blaze::DynamicMatrix<double> A2;
};
