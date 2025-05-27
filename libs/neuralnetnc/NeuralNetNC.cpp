#include <NeuralNetNC/NeuralNetNC.hpp>


NeuralNetNC::NeuralNetNC(
        unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories)
{
    W1 = nc::random::rand<float>({hidden_layer_size, num_features}) - 0.5f;
    b1 = nc::random::rand<float>({hidden_layer_size, 1}) - 0.5f;
    W2 = nc::random::rand<float>({categories, hidden_layer_size}) - 0.5f;
    b2 = nc::random::rand<float>({categories, 1}) - 0.5f;
    // W1({0, 3}, {0, 3}).print();
}

nc::NdArray<float> NeuralNetNC::ReLU(const nc::NdArray<float> &Z)
{
    return nc::maximum(Z, 0.0f);
}

nc::NdArray<float> NeuralNetNC::Softmax(const nc::NdArray<float> &Z)
{
    const auto e = nc::exp(Z);
    const auto s = nc::sum(e, nc::Axis::ROW);
    return e / s;
}

nc::NdArray<float> NeuralNetNC::forward_prop(const nc::NdArray<float> &X)
{
    Z1 = W1.dot(X) + b1;
    A1 = NeuralNetNC::ReLU(Z1);
    Z2 = W2.dot(A1) + b2;
    A2 = NeuralNetNC::Softmax(Z2);

    return A2;
}

nc::NdArray<int> NeuralNetNC::one_hot_encode(nc::NdArray<int> &Y)
{
    nc::NdArray<int> one_hot_Y = nc::zeros<int>(Y.numRows(), nc::uint32(Y.max().item()) + 1);
    for (int i = 0; i < Y.numRows(); i++)
    {
        one_hot_Y(i, Y(i, 0)) = 1;
    }
    // one_hot_Y.put(nc::arange(Y.size()), Y, 1);
    return one_hot_Y.transpose();
}

nc::NdArray<float> NeuralNetNC::deriv_ReLU(const nc::NdArray<float> &Z)
{
    return (Z > 0.0f).astype<float>();
}

void NeuralNetNC::back_prop(
        const nc::NdArray<float> &X, const nc::NdArray<int> &one_hot_Y, const float alpha)
{
    const float m = one_hot_Y.numRows();
    const auto dZ2 = A2 - one_hot_Y.astype<float>();
    dW2 = 1.0f / m * dZ2.dot(A1.transpose());
    db2 = (1.0f / m * nc::sum(dZ2)).item();

    const auto dZ1 = W2.transpose().dot(dZ2) * NeuralNetNC::deriv_ReLU(Z1);
    dW1 = 1.0f / m * dZ1.dot(X);
    db1 = (1.0f / m * nc::sum(dZ1)).item();

    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;
}

nc::NdArray<unsigned> NeuralNetNC::get_predictions(const nc::NdArray<float> &P)
{
    return nc::argmax(P, nc::Axis::ROW);
}

int NeuralNetNC::get_correct_prediction(const nc::NdArray<unsigned> &p, const nc::NdArray<int> &y)
{
    return nc::sum((p == y.astype<unsigned>().transpose()).astype<int>()).item();
}

float NeuralNetNC::get_accuracy(const int correct_prediction, const int size)
{
    return 1.0f * correct_prediction / size;
}

void NeuralNetNC::rnd_seed(const int seed)
{
    nc::random::seed(seed); // NOLINT(*-msc51-cpp)
}
