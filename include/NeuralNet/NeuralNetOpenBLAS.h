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

MatrixDouble *create_matrix(uint rows, uint cols)
{
    auto *mat = (MatrixDouble *) malloc(sizeof(MatrixDouble));
    mat->data = (double *) malloc(rows * cols * sizeof(double));
    return mat;
}

void free_matrix(MatrixDouble *mat)
{
    if (mat == NULL)
    {
        return;
    }
    if (mat->data != NULL)
    {
        free(mat->data);
        mat->data = NULL;
    }
    free(mat);
    mat = NULL;
}
