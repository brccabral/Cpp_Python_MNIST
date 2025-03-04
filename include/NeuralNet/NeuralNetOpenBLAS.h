#pragma once
#include <cblas.h>
#include <lapacke.h>
#include <stdlib.h>
#include <sys/types.h>
#include <immintrin.h>
#include <omp.h>

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
    uint rows;
    uint cols;
    double *data;
} MatrixDouble;

typedef struct NeuralNetOpenBLAS
{
    MatrixDouble *W1, *W2, *b1, *b2;
    MatrixDouble *Z1, *Z2, *A1, *A2, *A2ones, *A2result = NULL;
    uint num_inputs;
    uint num_hidden_layers;
    uint num_outputs;
} NeuralNetOpenBLAS;

MatrixDouble *create_matrix(uint rows, uint cols);
NeuralNetOpenBLAS *create_neuralnet_openblas(
        unsigned int num_features, unsigned int hidden_layer_size, unsigned int categories);
void fill_random_matrix(const MatrixDouble *mat, double offset);
void free_matrix(MatrixDouble *mat);
void free_neuralnet_openblas(NeuralNetOpenBLAS *nn);
void seed(size_t value);

MatrixDouble *one_hot_encode(const MatrixDouble *mat, uint column);
MatrixDouble *forward_prop(NeuralNetOpenBLAS *nn, const MatrixDouble *inputs);
void relu_ewise(const MatrixDouble *M);
void exp_ewise(const MatrixDouble *M);
void add_vector_to_matrix(const MatrixDouble *M, const MatrixDouble *V);
