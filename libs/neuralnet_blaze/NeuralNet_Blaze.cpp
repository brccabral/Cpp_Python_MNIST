#include <iostream>
#include <NeuralNet_Blaze/NeuralNet_Blaze.h>

NeuralNet_Blaze::NeuralNet_Blaze(int num_features, int hidden_layer_size, int categories)
{
    W1 = blaze::DynamicMatrix<double>(hidden_layer_size, num_features);
    blaze::randomize(W1, -0.5, 0.5);
    b1 = blaze::DynamicVector<double>(hidden_layer_size);
    blaze::randomize(b1, -0.5, 0.5);
    W2 = blaze::DynamicMatrix<double>(categories, hidden_layer_size);
    blaze::randomize(W2, -0.5, 0.5);
    b2 = blaze::DynamicVector<double>(categories);
    blaze::randomize(b2, -0.5, 0.5);
}

blaze::DynamicMatrix<double> NeuralNet_Blaze::one_hot_encode(const blaze::DynamicMatrix<double> &Z)
{
    auto o = blaze::DynamicMatrix<double>(Z.rows(), blaze::max(Z) + 1, 0);

    for (int r = 0; r < Z.rows() - 1; r++)
    {
        o(r, int(Z(r, 0))) = 1;
    }
    return o.transpose();
}
