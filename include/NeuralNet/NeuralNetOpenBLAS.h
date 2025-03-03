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

typedef struct VectorDouble
{
    uint rows;
    uint cols;
    double *data;
} VectorDouble;

typedef struct NeuralNetOpenBLAS
{
    MatrixDouble *W1, *W2;
    VectorDouble *b1, *b2;
} NeuralNetOpenBLAS;

MatrixDouble *create_matrix(uint rows, uint cols);
VectorDouble *create_vector(uint n);
void free_matrix(MatrixDouble *mat);
void free_vector(VectorDouble *vec);
