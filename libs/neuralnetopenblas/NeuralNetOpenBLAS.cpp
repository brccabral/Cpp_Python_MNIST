#include <cassert>
#include <cstring>
#include <NeuralNetOpenBLAS/NeuralNetOpenBLAS.h>
#include <immintrin.h>
#include <omp.h>

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

    omp_set_num_threads(std::max(omp_get_max_threads() - 2, 1));

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
    nn->A2 = NULL;
    nn->dW1 = NULL;
    nn->dW2 = NULL;
    nn->dZ1 = NULL;
    nn->A2ones = NULL;
    nn->A2sum = NULL;
    nn->predictions = NULL;
    return nn;
}

void free_matrix(MatrixDouble *mat)
{
    assert(mat);
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
    assert(nn);
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
        nn->A1 = NULL;
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
        nn->predictions = NULL;
    }
    if (nn->dW1 != NULL)
    {
        free_matrix(nn->dW1);
        nn->dW1 = NULL;
    }
    if (nn->dW2 != NULL)
    {
        free_matrix(nn->dW2);
        nn->dW2 = NULL;
    }
    if (nn->dZ1 != NULL)
    {
        free_matrix(nn->dZ1);
        nn->dZ1 = NULL;
    }
    free(nn);
    nn = NULL;
}

void fill_random_matrix(const MatrixDouble *mat, const double offset)
{
    assert(mat);
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
    assert(mat);
    assert(column < mat->cols);

    const double num_classes = mat->data[cblas_idmax(mat->rows, mat->data + column, mat->cols)] + 1;
    // one_hot_Y is transposed
    MatrixDouble *one_hot_Y = create_matrix(num_classes, mat->rows);
    if (one_hot_Y == NULL)
    {
        return NULL;
    }

    memset(one_hot_Y->data, 0, one_hot_Y->rows * one_hot_Y->cols * sizeof(double));

#pragma omp parallel for default(none) shared(mat, one_hot_Y, column)
    for (int i = 0; i < mat->rows; ++i)
    {
        const double value = (mat->data[i * mat->cols + column]);
        if (value >= 0 && value < one_hot_Y->rows)
        {
            one_hot_Y->data[uint32_t(one_hot_Y->cols * value + i)] = 1.0;
        }
    }
    return one_hot_Y;
}

void relu_ewise(const MatrixDouble *M)
{
    assert(M);
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
    assert(M);
    assert(V);
    assert(V->cols == 1);
    assert(M->rows == V->rows);

    // omp makes this loop slow
    // #pragma omp parallel for default(none) shared(M, V)
    for (int col = 0; col < M->cols; col++)
    {
        // Perform: M[row][col] += V[row] for each row
        // cblas_daxpy does: y = alpha * x + y, where x is V and y is M[:, col]
        cblas_daxpy(M->rows, 1.0, V->data, 1, &M->data[col], M->cols);
    }
}

void exp_ewise(const MatrixDouble *M)
{
    assert(M);

    // omp makes this loop slow
    // #pragma omp parallel for default(none) shared(M)
    for (int i = 0; i < M->rows * M->cols; i++)
    {
        M->data[i] = exp(M->data[i]);
    }
}

void matrix_div_vector_rwise(const MatrixDouble *M, const MatrixDouble *V)
{
    assert(M);
    assert(V);
    assert(V->cols == 1);
    assert(M->cols == V->rows);

    // omp makes this loop slow
    // #pragma omp parallel for default(none) shared(M, V)
    for (int col = 0; col < M->cols; ++col)
    {
        const double scale = 1.0 / V->data[col]; // Convert division into multiplication
        cblas_dscal(M->rows, scale, &M->data[col], M->cols);
    }
}

void create_aux(NeuralNetOpenBLAS *nn, const MatrixDouble *inputs)
{
    assert(nn);
    assert(inputs);
    assert(inputs->rows > 0);
    assert(inputs->cols > 0);
    if (nn->Z1 == NULL || nn->Z1->cols != inputs->rows)
    {
        if (nn->Z1)
        {
            free_matrix(nn->Z1);
        }
        nn->Z1 = create_matrix(nn->W1->rows, inputs->rows);

        if (nn->A1)
        {
            free_matrix(nn->A1);
        }
        nn->A1 = create_matrix(nn->W1->rows, inputs->rows);

        if (nn->A2)
        {
            free_matrix(nn->A2);
        }
        nn->A2 = create_matrix(nn->W2->rows, inputs->rows);

        if (nn->A2sum)
        {
            free_matrix(nn->A2sum);
        }
        nn->A2sum = create_matrix(nn->A2->cols, 1);

        if (nn->predictions)
        {
            free_matrix(nn->predictions);
        }
        nn->predictions = create_matrix(nn->A2->cols, 1);
        memset(nn->predictions->data, 0,
               nn->predictions->rows * nn->predictions->cols * sizeof(double));

        if (nn->dZ1)
        {
            free_matrix(nn->dZ1);
        }
        nn->dZ1 = create_matrix(nn->Z1->rows, nn->Z1->cols);

        if (nn->dW1)
        {
            free_matrix(nn->dW1);
        }
        nn->dW1 = create_matrix(nn->W1->rows, nn->W1->cols);

        if (nn->dW2)
        {
            free_matrix(nn->dW2);
        }
        nn->dW2 = create_matrix(nn->W2->rows, nn->W2->cols);

        if (nn->A2ones)
        {
            free_matrix(nn->A2ones);
        }
        nn->A2ones = create_matrix(nn->A2->rows, 1);
#pragma omp parallel for default(none) shared(nn)
        for (int i = 0; i < nn->A2ones->rows * nn->A2ones->cols; ++i)
        {
            nn->A2ones->data[i] = 1.0;
        }
    }
}

void forward_prop(NeuralNetOpenBLAS *nn, const MatrixDouble *inputs)
{
    // Z1 = W1.dot(X) + b1;
    // A1 = NeuralNetNC::ReLU(Z1);
    // Z2 = W2.dot(A1) + b2;
    // A2 = NeuralNetNC::Softmax(Z2);
    assert(nn);
    assert(inputs);
    assert(nn->W1->cols == inputs->cols); // inputs will be transposed in function call

    if (nn->Z1 == NULL || nn->Z1->cols != inputs->rows)
    {
        create_aux(nn, inputs);
    }

    // Z1 = W1.dot(Xt) + b1;
    multiply_ABt(nn->W1, inputs, nn->Z1);
    add_vector_to_matrix(nn->Z1, nn->b1);

    // A1 = NeuralNetNC::ReLU(Z1);
    memcpy(nn->A1->data, nn->Z1->data, nn->Z1->rows * nn->Z1->cols * sizeof(double));
    relu_ewise(nn->A1);

    // Z2 = W2.dot(A1) + b2;
    multiply_AB(nn->W2, nn->A1, nn->A2);
    add_vector_to_matrix(nn->A2, nn->b2);

    // A2 = NeuralNetNC::Softmax(Z2);
    exp_ewise(nn->A2);
    // A x VecOf1 = Sum(A, row)
    // multiplying a matrix A of a vector of 1's does the sum of A for each column
    cblas_dgemv(
            CblasRowMajor, CblasTrans, nn->A2->rows, nn->A2->cols, 1.0, nn->A2->data, nn->A2->cols,
            nn->A2ones->data, 1, 0.0, nn->A2sum->data, 1);
    matrix_div_vector_rwise(nn->A2, nn->A2sum);
}

void get_predictions(const NeuralNetOpenBLAS *nn)
{
    assert(nn);
    assert(nn->predictions);
    assert(nn->predictions->cols == 1);
    memset(nn->predictions->data, 0,
           nn->predictions->rows * nn->predictions->cols * sizeof(double));

#pragma omp parallel for default(none) shared(nn)
    for (int col = 0; col < nn->A2->cols; ++col)
    {
        const int index = cblas_idmax(nn->A2->rows, &nn->A2->data[col], nn->A2->cols);
        nn->predictions->data[col] = index % nn->A2->rows;
    }
}

uint get_correct_prediction(const NeuralNetOpenBLAS *nn, const MatrixDouble *labels)
{
    assert(nn);
    assert(labels);
    assert(nn->predictions->rows == labels->rows);
    assert(nn->predictions->cols == labels->cols);
    assert(nn->predictions->cols == 1);

    int correct_count = 0;

#pragma omp parallel for default(none) shared(nn, labels) reduction(+ : correct_count)
    for (int row = 0; row < nn->predictions->rows; ++row)
    {
        if (nn->predictions->data[row] == labels->data[row])
        {
            ++correct_count;
        }
    }

    return correct_count;
}

void deriv_ReLU_ewise(const MatrixDouble *M)
{
    assert(M);

    const int size = M->rows * M->cols;

    int i;
    // Process 4 elements at a time with SSE
    for (i = 0; i <= size - 4; i += 4)
    {
        // Load 4 values from input into a 128-bit register
        const __m256d data = _mm256_loadu_pd(&M->data[i]);

        // Compare the values with zero and set the result to 1.0 for positive, 0.0 for negative
        const auto mask = _mm256_cmp_pd(data, _mm256_setzero_pd(), _CMP_GT_OS);
        const __m256d result = _mm256_and_pd(mask, _mm256_set1_pd(1.0));

        // Store the result back into the output array
        _mm256_storeu_pd(&M->data[i], result);
    }

    // Process remaining elements (if any)
    for (; i < size; ++i)
    {
        M->data[i] = (M->data[i] > 0) ? 1.0f : 0.0f;
    }
}

// element-wise multiplication (Hadamard product) using AVX2 (double precision)
void product_ewise(const MatrixDouble *D, const MatrixDouble *Z)
{
    assert(D);
    assert(Z);
    assert(D->rows == Z->rows);
    assert(D->cols == Z->cols);

    const int total_elements = D->rows * D->cols;

    // Process 4 elements at a time using AVX2 (256-bit registers hold 4 doubles)
    int i = 0;
    for (; i <= total_elements - 4; i += 4)
    {
        const __m256d d_vals = _mm256_loadu_pd(&D->data[i]); // Load 4 elements of D
        const __m256d z_vals = _mm256_loadu_pd(&Z->data[i]); // Load 4 elements of Z
        const __m256d res_vals = _mm256_mul_pd(d_vals, z_vals); // NOLINT Multiply element-wise
        _mm256_storeu_pd(&D->data[i], res_vals); // Store the result
    }

    // Handle the remaining elements if not a multiple of 4
    for (; i < total_elements; i++)
    {
        D->data[i] = D->data[i] * Z->data[i];
    }
}

void subtract_scalar(const MatrixDouble *M, const double scalar)
{
    assert(M);

    const int size = M->rows * M->cols;

    // Load the scalar value into an AVX register. Since SIMD works on 256-bit registers,
    // we need to load 4 doubles at a time.
    const __m256d scalar_vector = _mm256_set1_pd(scalar);

    // Iterate over the matrix, processing 4 elements at a time
    int i;
    for (i = 0; i <= size - 4; i += 4)
    {
        // Load 4 elements from the matrix into an AVX register
        const __m256d mat_vector = _mm256_loadu_pd(&M->data[i]);

        // Subtract scalar_vector from mat_vector
        const __m256d result = _mm256_sub_pd(mat_vector, scalar_vector); // NOLINT

        // Store the result back into the matrix
        _mm256_storeu_pd(&M->data[i], result);
    }

    for (; i < size; ++i)
    {
        M->data[i] = M->data[i] - scalar;
    }
}

void back_prop(
        NeuralNetOpenBLAS *nn, const MatrixDouble *inputs, const MatrixDouble *one_hot_Y,
        const double alpha)
{
    assert(nn);
    assert(inputs);
    assert(one_hot_Y);
    assert(nn->W1->cols == inputs->cols);
    assert(nn->W2->rows == one_hot_Y->rows);
    if (nn->Z1 == NULL || nn->Z1->cols != inputs->rows)
    {
        create_aux(nn, inputs);
    }
    const int y_size = one_hot_Y->cols;

    // const Eigen::MatrixXf dZ2 = A2 - one_hot_Y;
    // reuse A2 as dZ2
    // A2 = categ x images
    // one_hot = categ, images
    cblas_daxpy(nn->A2->rows * nn->A2->cols, -1.0f, one_hot_Y->data, 1, nn->A2->data, 1);

    // dW2 = dZ2 * A1.transpose() / y_size;
    // dZ2/A2 = categ, images
    // A1 = hidden, images
    // A1_t = images, hidden
    // dW2 = categ, hidden
    multiply_ABt(nn->A2, nn->A1, nn->dW2);
    cblas_dscal(nn->dW2->rows * nn->dW2->cols, 1.0 / y_size, nn->dW2->data, 1);

    // db2 = dZ2.sum() / y_size;
    const double db2 = cblas_dsum(nn->A2->rows * nn->A2->cols, nn->A2->data, 1) / y_size;

    // const Eigen::MatrixXf dZ1 = (W2.transpose() * dZ2).cwiseProduct(deriv_ReLU(Z1));
    // w2 categ, hidden
    // dZ2 categ, images
    // dZ1 hidden, images
    // Z1 hidden, images
    deriv_ReLU_ewise(nn->Z1);
    multiply_AtB(nn->W2, nn->A2, nn->dZ1);
    product_ewise(nn->dZ1, nn->Z1);


    // dW1 = dZ1 * X / y_size;
    // dZ1 hidden, images
    // X images, features
    // dW1 hidden, features
    multiply_AB(nn->dZ1, inputs, nn->dW1);
    cblas_dscal(nn->dW1->rows * nn->dW1->cols, 1.0 / y_size, nn->dW1->data, 1);

    // db1 = dZ1.sum() / y_size;
    const double db1 = cblas_dsum(nn->dZ1->rows * nn->dZ1->cols, nn->dZ1->data, 1) / y_size;

    // W1 = W1 - dW1 * alpha;
    // W1 hidden, features
    // dW1 hidden, features
    cblas_daxpy(nn->W1->rows * nn->W1->cols, -alpha, nn->dW1->data, 1, nn->W1->data, 1);

    // b1 = b1.array() - db1 * alpha;
    subtract_scalar(nn->b1, db1 * alpha);

    // W2 = W2 - dW2 * alpha;
    cblas_daxpy(nn->W2->rows * nn->W2->cols, -alpha, nn->dW2->data, 1, nn->W2->data, 1);

    // b2 = b2.array() - db2 * alpha;
    subtract_scalar(nn->b2, db2 * alpha);
}

void multiply_AB(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result)
{
    assert(A);
    assert(B);
    assert(result);
    assert(A->cols == B->rows);
    assert(A->rows == result->rows);
    assert(B->cols == result->cols);
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, A->rows, B->cols, A->cols, 1.0, A->data,
            A->cols, B->data, B->cols, 0.0, result->data, result->cols);
}

void multiply_AtB(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result)
{
    assert(A);
    assert(B);
    assert(result);
    assert(A->rows == B->rows);
    assert(A->cols == result->rows);
    assert(B->cols == result->cols);
    cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans, A->cols, B->cols, A->rows, 1.0, A->data,
            A->cols, B->data, B->cols, 0.0, result->data, result->cols);
}

void multiply_ABt(const MatrixDouble *A, const MatrixDouble *B, const MatrixDouble *result)
{
    assert(A);
    assert(B);
    assert(result);
    assert(A->cols == B->cols);
    assert(A->rows == result->rows);
    assert(B->rows == result->cols);
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, A->rows, B->rows, A->cols, 1.0, A->data,
            A->cols, B->data, B->cols, 0.0, result->data, result->cols);
}
