#include <iostream>
#include <NeuralNet_Blaze/NeuralNet_Blaze.h>

NeuralNet_Blaze::NeuralNet_Blaze(
        const int num_features, const int hidden_layer_size, const int categories)
{
    W1 = blaze::rand<blaze::DynamicMatrix<double>>(hidden_layer_size, num_features) - 0.5;
    b1 = blaze::rand<blaze::DynamicVector<double>>(hidden_layer_size) - 0.5;
    W2 = blaze::rand<blaze::DynamicMatrix<double>>(categories, hidden_layer_size) - 0.5;
    b2 = blaze::rand<blaze::DynamicVector<double>>(categories) - 0.5;
}

blaze::DynamicMatrix<double> NeuralNet_Blaze::forward_prop(const blaze::DynamicMatrix<double> &X)
{
    Z1 = W1 * X;
    Z1 += expand(b1, Z1.columns());
    A1 = ReLU(Z1);

    Z2 = W2 * A1;
    Z2 += expand(b2, Z2.columns());
    A2 = Softmax(Z2);

    return A2;
}

void NeuralNet_Blaze::back_prop(
        const blaze::DynamicMatrix<double> &X, const blaze::DynamicMatrix<double> &target,
        const double alpha)
{
    const auto y_size = target.rows();

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

blaze::DynamicMatrix<double> NeuralNet_Blaze::one_hot_encode(const blaze::DynamicVector<double> &Z)
{
    blaze::DynamicMatrix<double> o = blaze::zero<double>(Z.size(), blaze::max(Z) + 1);

    for (int r = 0; r < Z.size() - 1; r++)
    {
        o(r, int(Z[r])) = 1;
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

blaze::DynamicMatrix<double> NeuralNet_Blaze::Softmax(const blaze::DynamicMatrix<double> &Z)
{
    blaze::DynamicMatrix<float> result = Z;

    // Subtract column-wise max from each row
    for (size_t i = 0; i < Z.columns(); ++i)
    {
        const double maxVal = blaze::max(blaze::column(Z, i));
        blaze::column(result, i) -= maxVal;
    }

    // Exponentiate
    result = blaze::exp(result);

    // Normalize each column by the column sum
    for (size_t i = 0; i < result.columns(); ++i)
    {
        const double sumVal = blaze::sum(blaze::column(result, i));
        blaze::column(result, i) /= sumVal;
    }

    return result;
}

blaze::DynamicVector<double>
NeuralNet_Blaze::get_predictions(const blaze::DynamicMatrix<double> &A2)
{
    blaze::DynamicVector<double> p = blaze::zero<double>(A2.columns());
    for (int c = 0; c < A2.columns(); c++)
    {
        p[c] = blaze::argmax(blaze::column(A2, c));
    }
    return p;
}

double NeuralNet_Blaze::get_correct_prediction(
        const blaze::DynamicVector<double> &predictions, const blaze::DynamicVector<double> &Y)
{
    blaze::DynamicVector<double> result(Y.size());
    for (int c = 0; c < Y.size(); c++)
    {
        result[c] = predictions[c] == Y[c] ? 1.0 : 0.0;
    }
    return blaze::sum(result);
}

double NeuralNet_Blaze::get_accuracy(const double correct_prediction, const size_t size)
{
    return correct_prediction / size;
}
