#include <iostream>
#include <NeuralNet_Blaze/NeuralNet_Blaze.h>

NeuralNet_Blaze::NeuralNet_Blaze(int num_features, int hidden_layer_size, int categories)
{
    W1 = blaze::DynamicMatrix<double>(hidden_layer_size, num_features);
    blaze::randomize(W1, -0.5, 0.5);
    std::cout << W1 << std::endl;
}
