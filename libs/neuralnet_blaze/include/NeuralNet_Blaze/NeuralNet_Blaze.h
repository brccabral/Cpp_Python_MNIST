#pragma once

#include <blaze/Blaze.h>

class NeuralNet_Blaze
{
public:

    NeuralNet_Blaze(int num_features, int hidden_layer_size, int categories);

    blaze::DynamicMatrix<double> W1;
    blaze::DynamicMatrix<double> b1;
    blaze::DynamicMatrix<double> W2;
    blaze::DynamicMatrix<double> b2;
    blaze::DynamicMatrix<double> Z1;
    blaze::DynamicMatrix<double> A1;
    blaze::DynamicMatrix<double> Z2;
    blaze::DynamicMatrix<double> A2;
};
