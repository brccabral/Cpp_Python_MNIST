#include <NeuralNet/NeuralNet.hpp>
#include <NeuralNet/NeuralNetNC.hpp>


NeuralNetNC::NeuralNetNC(unsigned int num_features,
                     unsigned int hidden_layer_size,
                     unsigned int categories)
{
    W1 = nc::random::rand<float>({hidden_layer_size, num_features}) - 0.5f;
    b1 = nc::random::rand<float>({hidden_layer_size, 1}) - 0.5f;
    W2 = nc::random::rand<float>({categories, hidden_layer_size}) - 0.5f;
    b2 = nc::random::rand<float>({categories, 1}) - 0.5f;
}

nc::NdArray<float> NeuralNetNC::ReLU(nc::NdArray<float> &Z)
{
    return nc::maximum(Z, 0.0f);
}

nc::NdArray<float> NeuralNetNC::Softmax(nc::NdArray<float> &Z)
{
    return nc::exp(Z) / nc::sum(nc::exp(Z));
}

nc::NdArray<float> NeuralNetNC::forward_prop(nc::NdArray<float> &X)
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
    one_hot_Y.print();
    return one_hot_Y.transpose();
}

nc::NdArray<bool> NeuralNetNC::deriv_ReLU(nc::NdArray<float> &Z)
{
    return Z > 0.0f;
}

// void NeuralNetNC::back_prop(
//     nc::NdArray<float> &X,
//     nc::NdArray<float> &Y,
//     nc::NdArray<float> &one_hot_Y,
//     float alpha)
// {
//     int y_size = Y.rows();
//
//     nc::NdArray<float> dZ2 = A2 - one_hot_Y;
//     dW2 = dZ2 * A1.transpose() / y_size;
//     db2 = dZ2.sum() / y_size;
//
//     nc::NdArray<float> dZ1 = (W2.transpose() * dZ2).cwiseProduct(deriv_ReLU(Z1));
//     dW1 = dZ1 * X.transpose() / y_size;
//     db1 = dZ1.sum() / y_size;
//
//     W1 = W1 - dW1 * alpha;
//     b1 = b1.array() - db1 * alpha;
//     W2 = W2 - dW2 * alpha;
//     b2 = b2.array() - db2 * alpha;
// }
//
// nc::NdArray<float> NeuralNetNC::get_predictions(nc::NdArray<float> &P)
// {
//     nc::NdArray<float> p = nc::NdArray<float>::Zero(P.cols()).array() - 1;
//     Eigen::Index maxIndex;
//     for (int c = 0; c < P.cols(); c++)
//     {
//         P.col(c).maxCoeff(&maxIndex);
//         p(c) = maxIndex;
//     }
//     return p;
// }
//
// int NeuralNetNC::get_correct_prediction(nc::NdArray<float> &p, nc::NdArray<float> &y)
// {
//     Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> e = p.cwiseEqual(y);
//     Eigen::VectorXi e_int = e.unaryExpr([](const bool x)
//                                         { return x ? 1 : 0; });
//     return e_int.sum();
// }
//
// float NeuralNetNC::get_accuracy(int correct_prediction, int size)
// {
//     return 1.0f * correct_prediction / size;
// }
