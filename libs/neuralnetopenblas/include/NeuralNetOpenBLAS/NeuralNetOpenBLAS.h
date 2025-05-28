#pragma once
#include <cstdint>
#include <cblas.h>
#if !defined(_MSC_VER)
#include <lapacke.h>
#endif
#include <sys/types.h>

#define DESCRIBE_MATRIX(m) printf("%s: %d,%d\n", #m, (m)->rows, (m)->cols)

#define DESCRIBE_NN(nn)                                                                            \
    printf("%s\n", (#nn));                                                                         \
    DESCRIBE_MATRIX((nn)->W1);                                                                     \
    DESCRIBE_MATRIX((nn)->b1);                                                                     \
    DESCRIBE_MATRIX((nn)->W2);                                                                     \
    DESCRIBE_MATRIX((nn)->b2);                                                                     \
    printf("num_inputs %d, num_hidden_layers %d, num_outputs %d\n", (nn)->num_inputs,              \
           (nn)->num_hidden_layers, (nn)->num_outputs)

typedef struct MatrixDouble
{
    uint32_t rows;
    uint32_t cols;
    double *data;
} MatrixDouble;

typedef struct NeuralNetOpenBLAS
{
    // layers
    MatrixDouble *W1 = NULL, *W2 = NULL, *b1 = NULL, *b2 = NULL;
    // derivatives
    MatrixDouble *Z1 = NULL, *A1 = NULL, *A2 = NULL;
    MatrixDouble *dW1 = NULL, *dW2 = NULL, *dZ1 = NULL;
    MatrixDouble *db1 = NULL, *db2 = NULL;
    // auxiliaries
    MatrixDouble *A2ones = NULL, *A2sum = NULL, *predictions = NULL;

    uint32_t num_inputs = 0;
    uint32_t num_hidden_layers = 0;
    uint32_t num_outputs = 0;
} NeuralNetOpenBLAS;

MatrixDouble *create_matrix(uint32_t rows, uint32_t cols);
NeuralNetOpenBLAS *create_neuralnet_openblas(
        unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories);
void fill_random_matrix(const MatrixDouble *mat, double offset);
void free_matrix(MatrixDouble *mat);
void free_neuralnet_openblas(NeuralNetOpenBLAS *nn);
void nn_seed(size_t value);

MatrixDouble *one_hot_encode(const MatrixDouble *mat, uint32_t column);
void forward_prop(NeuralNetOpenBLAS *nn, const MatrixDouble *inputs);
void relu_ewise(const MatrixDouble *M);
void deriv_ReLU_ewise(const MatrixDouble *M);
void exp_ewise(const MatrixDouble *M);
void add_vector_to_matrix(const MatrixDouble *M, const MatrixDouble *V);
void get_predictions(const NeuralNetOpenBLAS *nn);
uint32_t get_correct_prediction(const NeuralNetOpenBLAS *nn, const MatrixDouble *labels);
void product_ewise(const MatrixDouble *D, const MatrixDouble *Z);
void subtract_scalar(const MatrixDouble *M, double scalar);
void back_prop(
        NeuralNetOpenBLAS *nn, const MatrixDouble *inputs, const MatrixDouble *one_hot_Y,
        double alpha);
void multiply_AB(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result);
void multiply_AtB(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result);
void multiply_ABt(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result);
