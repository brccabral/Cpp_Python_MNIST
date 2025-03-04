#include "NeuralNet/NeuralNetXT.hpp"
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>


NeuralNetXT::NeuralNetXT(
        unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories)
{
    W1 = xt::random::rand<float>({hidden_layer_size, num_features}) - 0.5f;
    b1 = xt::random::rand<float>({hidden_layer_size, (unsigned int) 1}) - 0.5f;
    W2 = xt::random::rand<float>({categories, hidden_layer_size}) - 0.5f;
    b2 = xt::random::rand<float>({categories, (unsigned int) 1}) - 0.5f;
}

xt::xarray<float> NeuralNetXT::ReLU(const xt::xarray<float> &Z)
{
    return xt::maximum(Z, 0.0f);
}

xt::xarray<float> NeuralNetXT::Softmax(const xt::xarray<float> &Z)
{
    const auto e = xt::exp(Z);
    const auto s = xt::sum(e, {0});
    return e / s;
}

xt::xarray<float> NeuralNetXT::forward_prop(const xt::xarray<float> &X)
{
    Z1 = xt::linalg::dot(W1, X) + b1;
    A1 = NeuralNetXT::ReLU(Z1);
    Z2 = xt::linalg::dot(W2, A1) + b2;
    A2 = NeuralNetXT::Softmax(Z2);

    return A2;
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

xt::xarray<float> NeuralNetXT::deriv_ReLU(const xt::xarray<float> &Z)
{
    return xt::cast<float>(Z > 0.0f);
}

void NeuralNetXT::back_prop(
        const xt::xarray<float> &X, const xt::xarray<int> &Y, const xt::xarray<int> &one_hot_Y,
        const float alpha)
{
    const auto m = (float) Y.size();
    const auto dZ2 = A2 - xt::cast<float>(one_hot_Y);
    dW2 = 1.0f / m * xt::linalg::dot(dZ2, xt::transpose(A1));
    db2 = (1.0f / m * xt::sum(dZ2))();

    const auto dZ1 = xt::linalg::dot(xt::transpose(W2), dZ2) * NeuralNetXT::deriv_ReLU(Z1);
    dW1 = 1.0f / m * xt::linalg::dot(dZ1, X);
    db1 = (1.0f / m * xt::sum(dZ1))();

    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;
}

xt::xarray<int> NeuralNetXT::get_predictions(const xt::xarray<float> &P)
{
    return xt::argmax(P, 0);
}

int NeuralNetXT::get_correct_prediction(const xt::xarray<int> &p, const xt::xarray<int> &y)
{
    return xt::sum(xt::equal(p, y))();
}

float NeuralNetXT::get_accuracy(const int correct_prediction, const int size)
{
    return 1.0f * correct_prediction / size;
}

void NeuralNetXT::rnd_seed(const int seed)
{
    xt::random::seed(seed); // NOLINT(*-msc51-cpp)
}
