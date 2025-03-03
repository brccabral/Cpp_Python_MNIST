#pragma once
#include <cblas.h>
#include <stdlib.h>
#include <sys/types.h>

typedef struct MatrixDouble
{
    uint rows;
    uint cols;
    double *data;
} MatrixDouble;

typedef struct NeuralNetOpenBLAS
{
    MatrixDouble *W1, *W2, *b1, *b2;
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
