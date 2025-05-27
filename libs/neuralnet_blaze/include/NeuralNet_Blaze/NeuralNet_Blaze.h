#pragma once

#include <blaze/Blaze.h>

class NeuralNet_Blaze
{
public:

    NeuralNet_Blaze(int num_features, int hidden_layer_size, int categories);

    blaze::DynamicMatrix<double> forward_prop(const blaze::DynamicMatrix<double> &X);
    void back_prop(
            const blaze::DynamicMatrix<double> &X, const blaze::DynamicMatrix<double> &target,
            double alpha);

    static blaze::DynamicMatrix<double> one_hot_encode(const blaze::DynamicVector<double> &Z);
    static blaze::DynamicMatrix<double> ReLU(const blaze::DynamicMatrix<double> &Z);
    static blaze::DynamicMatrix<double> deriv_ReLU(const blaze::DynamicMatrix<double> &Z);
    static blaze::DynamicVector<double> get_predictions(const blaze::DynamicMatrix<double> &A2);
    static double get_correct_prediction(
            const blaze::DynamicVector<double> &predictions, const blaze::DynamicVector<double> &Y);
    static double get_accuracy(double correct_prediction, size_t size);

    blaze::DynamicMatrix<double> W1;
    blaze::DynamicVector<double> b1;
    blaze::DynamicMatrix<double> W2;
    blaze::DynamicVector<double> b2;
    blaze::DynamicMatrix<double> Z1;
    blaze::DynamicMatrix<double> A1;
    blaze::DynamicMatrix<double> Z2;
    blaze::DynamicMatrix<double> A2;
};
