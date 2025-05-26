#include <iostream>
#include <NeuralNet_Blaze/NeuralNet_Blaze.h>

NeuralNet_Blaze::NeuralNet_Blaze(
        const int num_features, const int hidden_layer_size, const int categories)
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

blaze::DynamicMatrix<double> NeuralNet_Blaze::forward_prop(const blaze::DynamicMatrix<double> &X)
{
    Z1 = W1 * X;
    Z1 += expand(b1, Z1.columns());
    A1 = ReLU(Z1);

    Z2 = W2 * A1;
    Z2 += expand(b2, Z2.columns());
    A2 = blaze::softmax(Z2);

    return A2;
}

void NeuralNet_Blaze::back_prop(
        const blaze::DynamicMatrix<double> &X, const blaze::DynamicMatrix<double> &target,
        const double alpha)
{
    const auto y_size = target.columns();

    const blaze::DynamicMatrix<double> dZ2 = (A2 - target) / y_size;
    const blaze::DynamicMatrix<double> dW2 = dZ2 * blaze::trans(A1);
    const blaze::DynamicVector<double> db2 = blaze::sum<blaze::rowwise>(dZ2);

    const blaze::DynamicMatrix<double> dZ1 = (blaze::trans(W2) * dZ2) % deriv_ReLU(Z1);
    const blaze::DynamicMatrix<double> dW1 = dZ1 * X;
    const blaze::DynamicVector<double> db1 = blaze::sum<blaze::rowwise>(dZ1);

    W1 -= (dW1 * alpha);
    W2 -= (dW2 * alpha);
    b1 -= (db1 * alpha);
    b2 -= (db2 * alpha);
}

blaze::DynamicMatrix<double> NeuralNet_Blaze::one_hot_encode(const blaze::DynamicMatrix<double> &Z)
{
    blaze::DynamicMatrix<double> o = blaze::zero<double>(Z.rows(), blaze::max(Z) + 1);

    for (int r = 0; r < Z.rows() - 1; r++)
    {
        o(r, int(Z(r, 0))) = 1;
    }
    return o.transpose();
}

blaze::DynamicMatrix<double> NeuralNet_Blaze::ReLU(const blaze::DynamicMatrix<double> &Z)
{
    return blaze::max(Z, 0.0);
}

blaze::DynamicMatrix<double> NeuralNet_Blaze::deriv_ReLU(const blaze::DynamicMatrix<double> &Z)
{
    return blaze::map(Z, [](const double z) { return z > 0.0 ? 1.0 : 0.0; });
}
