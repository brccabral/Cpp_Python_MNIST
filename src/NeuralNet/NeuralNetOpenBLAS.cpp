#include <NeuralNet/NeuralNetOpenBLAS.h>

MatrixDouble *create_matrix(const uint rows, const uint cols)
{
    auto *mat = (MatrixDouble *) malloc(sizeof(MatrixDouble));
    mat->data = (double *) malloc(rows * cols * sizeof(double));
    return mat;
}

VectorDouble *create_vector(const uint n)
{
    auto *vec = (VectorDouble *) malloc(sizeof(VectorDouble));
    vec->data = (double *) malloc(n * sizeof(double));
    return vec;
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

void free_vector(VectorDouble *vec)
{
    if (vec == NULL)
    {
        return;
    }
    if (vec->data != NULL)
    {
        free(vec->data);
        vec->data = NULL;
    }
    free(vec);
    vec = NULL;
}
