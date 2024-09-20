#include <NeuralNet/NeuralNet.hpp>
#include <unsupported/Eigen/MatrixFunctions>

NeuralNet::NeuralNet(const int num_features, const int hidden_layer_size, const int categories)
{
    // Random generates [-1:1]. Numpy is [0:1]
    W1 = Eigen::MatrixXf::Random(hidden_layer_size, num_features);
    W1 = W1.array() / 2.0f;
    b1 = Eigen::VectorXf::Random(hidden_layer_size);
    b1 = b1.array() / 2.0f;
    W2 = Eigen::MatrixXf::Random(categories, hidden_layer_size);
    W2 = W2.array() / 2.0f;
    b2 = Eigen::VectorXf::Random(categories, 1);
    b2 = b2.array() / 2.0f;
}

Eigen::MatrixXf NeuralNet::ReLU(const Eigen::MatrixXf &Z)
{
    return Z.cwiseMax(0);
}

Eigen::MatrixXf NeuralNet::Softmax(Eigen::MatrixXf &Z)
{
    Eigen::MatrixXf e = Z.array().exp();
    Eigen::MatrixXf s = e.colwise().sum();
    for (int c = 0; c < e.cols(); c++)
    {
        e.col(c) = e.col(c) / s(c);
    }
    return e;
}

Eigen::MatrixXf NeuralNet::forward_prop(const Eigen::MatrixXf &X)
{
    Z1 = W1 * X;
    for (int c = 0; c < Z1.cols(); c++)
    {
        Z1.col(c) = Z1.col(c) + b1;
    }
    A1 = ReLU(Z1);

    Z2 = W2 * A1;
    for (int c = 0; c < Z2.cols(); c++)
    {
        Z2.col(c) = Z2.col(c) + b2;
    }
    A2 = Softmax(Z2);

    return A2;
}

Eigen::MatrixXf NeuralNet::one_hot_encode(Eigen::VectorXf &Z)
{
    Eigen::MatrixXf o = Eigen::MatrixXf::Zero(Z.rows(), Z.maxCoeff() + 1);

    for (int r = 0; r < Z.rows() - 1; r++)
    {
        o(r, int(Z(r))) = 1;
    }
    return o.transpose();
}

Eigen::MatrixXf NeuralNet::deriv_ReLU(Eigen::MatrixXf &Z)
{
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> b2 = (Z.array() > 0);
    return b2.unaryExpr([](const bool x) { return x ? 1.0f : 0.0f; });
}

void NeuralNet::back_prop(
        const Eigen::MatrixXf &X, const Eigen::VectorXf &Y, const Eigen::MatrixXf &one_hot_Y,
        const float alpha)
{
    const int y_size = Y.rows();

    const Eigen::MatrixXf dZ2 = A2 - one_hot_Y;
    dW2 = dZ2 * A1.transpose() / y_size;
    db2 = dZ2.sum() / y_size;

    const Eigen::MatrixXf dZ1 = (W2.transpose() * dZ2).cwiseProduct(deriv_ReLU(Z1));
    dW1 = dZ1 * X.transpose() / y_size;
    db1 = dZ1.sum() / y_size;

    W1 = W1 - dW1 * alpha;
    b1 = b1.array() - db1 * alpha;
    W2 = W2 - dW2 * alpha;
    b2 = b2.array() - db2 * alpha;
}

Eigen::VectorXf NeuralNet::get_predictions(Eigen::MatrixXf &P)
{
    Eigen::VectorXf p = Eigen::VectorXf::Zero(P.cols()).array() - 1;
    Eigen::Index maxIndex;
    for (int c = 0; c < P.cols(); c++)
    {
        P.col(c).maxCoeff(&maxIndex);
        p(c) = maxIndex;
    }
    return p;
}

int NeuralNet::get_correct_prediction(const Eigen::VectorXf &p, const Eigen::VectorXf &y)
{
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> e = p.cwiseEqual(y);
    const Eigen::VectorXi e_int = e.unaryExpr([](const bool x) { return x ? 1 : 0; });
    return e_int.sum();
}

float NeuralNet::get_accuracy(const int correct_prediction, const int size)
{
    return 1.0f * correct_prediction / size;
}
