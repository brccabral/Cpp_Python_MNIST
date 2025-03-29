#pragma once

#include <Eigen/Dense>

class NeuralNet
{
public:

    // layers
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::MatrixXf W2;
    Eigen::VectorXf b2;

    // back prop
    Eigen::MatrixXf Z1;
    Eigen::MatrixXf A1;
    Eigen::MatrixXf Z2;
    Eigen::MatrixXf A2;

    // gradients
    Eigen::MatrixXf dW1;
    float db1{};
    Eigen::MatrixXf dW2;
    float db2{};

public:

    NeuralNet(int num_features, int hidden_layer_size, int categories);

    static Eigen::MatrixXf ReLU(const Eigen::MatrixXf &Z);

    static Eigen::MatrixXf Softmax(Eigen::MatrixXf &Z);

    Eigen::MatrixXf forward_prop(const Eigen::MatrixXf &X);

    static Eigen::MatrixXf one_hot_encode(Eigen::VectorXf &Z);

    static Eigen::MatrixXf deriv_ReLU(Eigen::MatrixXf &Z);

    void back_prop(const Eigen::MatrixXf &X, const Eigen::MatrixXf &target, float alpha);

    static Eigen::VectorXf get_predictions(Eigen::MatrixXf &P);

    static int get_correct_prediction(const Eigen::VectorXf &p, const Eigen::VectorXf &y);

    static float get_accuracy(int correct_prediction, int size);
};
