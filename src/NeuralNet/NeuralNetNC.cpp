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
//
// nc::NdArray<float> NeuralNetNC::ReLU(nc::NdArray<float> &Z)
// {
//     return Z.cwiseMax(0);
// }
//
// nc::NdArray<float> NeuralNetNC::Softmax(nc::NdArray<float> &Z)
// {
//     nc::NdArray<float> e = Z.array().exp();
//     nc::NdArray<float> s = e.colwise().sum();
//     for (int c = 0; c < e.cols(); c++)
//     {
//         e.col(c) = e.col(c) / s(c);
//     }
//     return e;
// }
//
// nc::NdArray<float> NeuralNetNC::forward_prop(nc::NdArray<float> &X)
// {
//     Z1 = W1 * X;
//     for (int c = 0; c < Z1.cols(); c++)
//     {
//         Z1.col(c) = Z1.col(c) + b1;
//     }
//     A1 = ReLU(Z1);
//
//     Z2 = W2 * A1;
//     for (int c = 0; c < Z2.cols(); c++)
//     {
//         Z2.col(c) = Z2.col(c) + b2;
//     }
//     A2 = Softmax(Z2);
//
//     return A2;
// }

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

// nc::NdArray<float> NeuralNetNC::deriv_ReLU(nc::NdArray<float> &Z)
// {
//     Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> b2 = (Z.array() > 0);
//     return b2.unaryExpr([](const bool x)
//                         { return x ? 1.0f : 0.0f; });
// }
//
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
