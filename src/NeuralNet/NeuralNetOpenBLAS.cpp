#include <NeuralNet/NeuralNetOpenBLAS.h>

MatrixDouble *create_matrix(const uint rows, const uint cols)
{
    auto *mat = (MatrixDouble *) malloc(sizeof(MatrixDouble));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double *) malloc(rows * cols * sizeof(double));
    return mat;
}

NeuralNetOpenBLAS *create_neuralnet_openblas(
        const unsigned int num_features, const unsigned int hidden_layer_size,
        const unsigned int categories)
{
    auto *nn = (NeuralNetOpenBLAS *) malloc(sizeof(NeuralNetOpenBLAS));
    nn->num_inputs = num_features;
    nn->num_hidden_layers = hidden_layer_size;
    nn->num_outputs = categories;
    nn->W1 = create_matrix(nn->num_hidden_layers, nn->num_inputs);
    nn->b1 = create_matrix(nn->num_hidden_layers, 1);
    nn->W2 = create_matrix(nn->num_outputs, nn->num_hidden_layers);
    nn->b2 = create_matrix(nn->num_outputs, 1);
    fill_random_matrix(nn->W1, -0.5);
    fill_random_matrix(nn->b1, -0.5);
    fill_random_matrix(nn->W2, -0.5);
    fill_random_matrix(nn->b2, -0.5);
    return nn;
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

void free_neuralnet_openblas(NeuralNetOpenBLAS *nn)
{
    if (nn == NULL)
    {
        return;
    }
    if (nn->W1 != NULL)
    {
        free_matrix(nn->W1);
    }
    if (nn->W2 != NULL)
    {
        free_matrix(nn->W2);
    }
    free(nn);
    nn = NULL;
}

void fill_random_matrix(const MatrixDouble *mat, const double offset)
{
    if (mat == NULL)
    {
        return;
    }
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->data[i] = drand48() + offset;
    }
}

void seed(const size_t value)
{
    srand48(value);
}
