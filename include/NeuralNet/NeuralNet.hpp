#pragma once

#include <eigen3/Eigen/Dense>

class NeuralNet
{
private:
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
    float db1;
    Eigen::MatrixXf dW2;
    float db2;

public:
    NeuralNet(int hidden_layer_size,
              int categories,
              int num_features);

    static Eigen::MatrixXf ReLU(Eigen::MatrixXf &Z);

    static Eigen::MatrixXf Softmax(Eigen::MatrixXf &Z);

    Eigen::MatrixXf forward_prop(Eigen::MatrixXf &X);

    static Eigen::MatrixXf one_hot_encode(Eigen::VectorXf &Z);

    static Eigen::MatrixXf deriv_ReLU(Eigen::MatrixXf &Z);

    void back_prop(
        Eigen::MatrixXf &X,
        Eigen::VectorXf &Y,
        Eigen::MatrixXf &one_hot_Y,
        float alpha);

    static Eigen::VectorXf get_predictions(Eigen::MatrixXf &P);

    static int get_correct_prediction(Eigen::VectorXf &p, Eigen::VectorXf &y);

    static float get_accuracy(int correct_prediction, int size);
};
