#include <cassert>
#include <cstring>
#include <NeuralNet/NeuralNetOpenBLAS.h>

MatrixDouble *create_matrix(const uint rows, const uint cols)
{
    assert(rows > 0 && cols > 0);

    auto *mat = (MatrixDouble *) malloc(sizeof(MatrixDouble));
    if (mat == NULL)
    {
        return NULL;
    }
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double *) malloc(rows * cols * sizeof(double));
    if (mat->data == NULL)
    {
        free(mat);
        return NULL;
    }
    return mat;
}

NeuralNetOpenBLAS *create_neuralnet_openblas(
        const unsigned int num_features, const unsigned int hidden_layer_size,
        const unsigned int categories)
{
    assert(num_features > 0 && hidden_layer_size > 0 && categories > 0);

    auto *nn = (NeuralNetOpenBLAS *) malloc(sizeof(NeuralNetOpenBLAS));
    if (nn == NULL)
    {
        return NULL;
    }
    nn->num_inputs = num_features;
    nn->num_hidden_layers = hidden_layer_size;
    nn->num_outputs = categories;
    nn->W1 = create_matrix(nn->num_hidden_layers, nn->num_inputs);
    if (nn->W1 == NULL)
    {
        free_neuralnet_openblas(nn);
        return NULL;
    }
    nn->b1 = create_matrix(nn->num_hidden_layers, 1);
    if (nn->b1 == NULL)
    {
        free_neuralnet_openblas(nn);
        return NULL;
    }
    nn->W2 = create_matrix(nn->num_outputs, nn->num_hidden_layers);
    if (nn->W2 == NULL)
    {
        free_neuralnet_openblas(nn);
        return NULL;
    }
    nn->b2 = create_matrix(nn->num_outputs, 1);
    if (nn->b2 == NULL)
    {
        free_neuralnet_openblas(nn);
        return NULL;
    }
    fill_random_matrix(nn->W1, -0.5);
    fill_random_matrix(nn->b1, -0.5);
    fill_random_matrix(nn->W2, -0.5);
    fill_random_matrix(nn->b2, -0.5);
    nn->Z1 = NULL;
    nn->A1 = NULL;
    nn->Z2 = NULL;
    nn->A2 = NULL;
    nn->A2ones = NULL;
    nn->A2sum = NULL;
    nn->predictions = NULL;
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
        nn->W1 = NULL;
    }
    if (nn->W2 != NULL)
    {
        free_matrix(nn->W2);
        nn->W2 = NULL;
    }
    if (nn->b1 != NULL)
    {
        free_matrix(nn->b1);
        nn->b1 = NULL;
    }
    if (nn->b2 != NULL)
    {
        free_matrix(nn->b2);
        nn->b2 = NULL;
    }
    if (nn->Z1 != NULL)
    {
        free_matrix(nn->Z1);
        nn->Z1 = NULL;
    }
    if (nn->A1 != NULL)
    {
        free_matrix(nn->A1);
    }
    if (nn->Z2 != NULL)
    {
        free_matrix(nn->Z2);
    }
    if (nn->A2 != NULL)
    {
        free_matrix(nn->A2);
        nn->A2 = NULL;
    }
    if (nn->A2ones != NULL)
    {
        free_matrix(nn->A2ones);
        nn->A2ones = NULL;
    }
    if (nn->A2sum != NULL)
    {
        free_matrix(nn->A2sum);
        nn->A2sum = NULL;
    }
    if (nn->predictions != NULL)
    {
        free_matrix(nn->predictions);
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

void nn_seed(const size_t value)
{
    srand48(value);
}

MatrixDouble *one_hot_encode(const MatrixDouble *mat, const uint column)
{
    if (mat == NULL)
    {
        return NULL;
    }
    assert(column < mat->cols);

    const uint32_t num_classes =
            mat->data[cblas_idmax(mat->rows, mat->data + column, mat->cols)] + 1;
    MatrixDouble *one_hot_Y = create_matrix(mat->rows, num_classes);

    memset(one_hot_Y->data, 0, mat->rows * num_classes * sizeof(int));

#pragma omp parallel for simd
    for (int i = 0; i < one_hot_Y->rows; ++i)
    {
        const double value = (mat->data[i * mat->cols + column]);
        if (value >= 0 && value < num_classes)
        {
            one_hot_Y->data[uint32_t(i * num_classes + value)] = 1.0;
        }
    }
    return one_hot_Y;
}

void relu_ewise(const MatrixDouble *M)
{
    if (M == NULL)
    {
        return;
    }
    const __m256d zero_vec = _mm256_setzero_pd(); // Load zero into AVX register

    const int size = M->rows * M->cols;

    int i;
    for (i = 0; i <= size - 4; i += 4)
    {
        __m256d val = _mm256_loadu_pd(&M->data[i]); // Load 4 doubles from M[i]
        val = _mm256_max_pd(val, zero_vec); // NOLINT Apply max(0, x) using AVX
        _mm256_storeu_pd(&M->data[i], val); // Store back the result
    }

    // Handle remaining elements (if size is not a multiple of 4)
    for (; i < size; i++)
    {
        if (M->data[i] < 0)
        {
            M->data[i] = 0;
        }
    }
}

// M is an i x j matrix
// V is a vector with size i, to be added to each column of M
void add_vector_to_matrix(const MatrixDouble *M, const MatrixDouble *V)
{
    if (M == NULL)
    {
        return;
    }
    if (V == NULL)
    {
        return;
    }
    assert(V->cols == 1);
    assert(M->rows == V->rows);

#pragma omp parallel for simd
    for (int col = 0; col < M->cols; col++)
    {
        // Perform: M[row][col] += V[row] for each row
        // cblas_daxpy does: y = alpha * x + y, where x is V and y is M[:, col]
        cblas_daxpy(M->rows, 1.0, V->data, 1, &M->data[col * M->rows], 1);
    }
}

void exp_ewise(const MatrixDouble *M)
{
    if (M == NULL)
    {
        return;
    }

#pragma omp parallel for simd
    for (int i = 0; i < M->rows * M->rows; i++)
    {
        M->data[i] = exp(M->data[i]);
    }
}

void matrix_div_vector_rwise(const MatrixDouble *M, const MatrixDouble *V)
{
    if (M == NULL)
    {
        return;
    }
    if (V == NULL)
    {
        return;
    }
    assert(V->cols == 1);
    assert(M->cols == V->rows);

#pragma omp parallel for simd
    for (int col = 0; col < M->cols; ++col)
    {
        const double scale = 1.0 / V->data[col]; // Convert division into multiplication
        cblas_dscal(M->rows, scale, &M->data[col * M->rows], 1); // Scale row i
    }
}

void forward_prop(NeuralNetOpenBLAS *nn, const MatrixDouble *inputs)
{
    // Z1 = W1.dot(X) + b1;
    // A1 = NeuralNetNC::ReLU(Z1);
    // Z2 = W2.dot(A1) + b2;
    // A2 = NeuralNetNC::Softmax(Z2);
    if (nn == NULL)
    {
        return;
    }
    if (inputs == NULL)
    {
        return;
    }

    assert(nn->W1->cols == inputs->cols); // inputs will be transposed in function call

    if (nn->Z1 == NULL || nn->Z1->cols != inputs->rows)
    {
        free_matrix(nn->Z1);
        free_matrix(nn->A1);
        free_matrix(nn->Z2);
        free_matrix(nn->A2);
        free_matrix(nn->A2ones);
        free_matrix(nn->A2sum);
        free_matrix(nn->predictions);
        nn->Z1 = create_matrix(nn->W1->rows, inputs->rows);
        nn->A1 = create_matrix(nn->W1->rows, inputs->rows);
        nn->Z2 = create_matrix(nn->W2->rows, inputs->rows);
        nn->A2 = create_matrix(nn->W2->rows, inputs->rows);
        nn->A2ones = create_matrix(nn->A2->cols, 1);
        nn->A2sum = create_matrix(nn->A2->cols, 1);
        nn->predictions = create_matrix(nn->A2->cols, 1);
#pragma omp parallel for simd
        for (int i = 0; i < nn->A2ones->rows * nn->A2ones->cols; ++i)
        {
            nn->A2ones->data[i] = 1.0;
        }
    }

    // Z1 = W1.dot(X) + b1;
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, nn->W1->rows, inputs->rows, inputs->cols, 1.0,
            nn->W1->data, nn->W1->cols, inputs->data, inputs->cols, 0.0, nn->Z1->data,
            nn->Z1->cols);
    add_vector_to_matrix(nn->Z1, nn->b1);

    // A1 = NeuralNetNC::ReLU(Z1);
    memcpy(nn->A1->data, nn->Z1->data, nn->Z1->rows * nn->Z1->cols * sizeof(double));
    relu_ewise(nn->A1);

    // Z2 = W2.dot(A1) + b2;
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, nn->W2->rows, nn->A1->cols, nn->A1->rows,
            1.0, nn->W2->data, nn->W2->cols, nn->A1->data, nn->A1->cols, 0.0, nn->Z2->data,
            nn->Z2->cols);
    add_vector_to_matrix(nn->Z2, nn->b2);

    // A2 = NeuralNetNC::Softmax(Z2);
    memcpy(nn->A2->data, nn->Z2->data, nn->Z2->rows * nn->Z2->cols * sizeof(double));
    exp_ewise(nn->A2);
    // A x VecOf1 = Sum(A, row)
    cblas_dgemv(
            CblasRowMajor, CblasNoTrans, nn->A2->rows, nn->A2->cols, 1.0, nn->A2->data,
            nn->A2->cols, nn->A2ones->data, 1, 0.0, nn->A2sum->data, 1);
    matrix_div_vector_rwise(nn->A2, nn->A2sum);
}

void get_predictions(NeuralNetOpenBLAS *nn)
{
    if (nn == NULL)
    {
        return;
    }
    assert(nn->predictions->cols == 1);

#pragma omp parallel for simd
    for (int row = 0; row < nn->A2->rows; ++row)
    {
        const int index = cblas_idmax(nn->A2->cols, &nn->A2->data[row * nn->A2->cols], 1);
        nn->predictions->data[row] = index;
    }
}

uint get_correct_prediction(NeuralNetOpenBLAS *nn, MatrixDouble *labels)
{
    if (nn == NULL)
    {
        return 0;
    }
    if (labels == NULL)
    {
        return 0;
    }
    assert(nn->predictions->rows == labels->rows);
    assert(nn->predictions->cols == labels->cols);
    assert(nn->predictions->cols == 1);

    int correct_count = 0;

#pragma omp parallel for simd reduction(+ : correct_count)
    for (int row = 0; row < nn->predictions->rows; ++row)
    {
        if (nn->predictions->data[row] == labels->data[row])
        {
            correct_count++;
        }
    }

    return correct_count;
}
