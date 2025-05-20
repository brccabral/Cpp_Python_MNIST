#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>
#include <climits>
#include <cfloat>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);

bool init_numpy()
{
    import_array1(false);
    return true;
}

CNumpy::CNumpy()
{
    Py_Initialize();
    if (!init_numpy())
    {
        throw std::invalid_argument("Could not init numpy.");
    }
}

CNumpy::~CNumpy()
{
    Py_Finalize();
}

CNdArray CNumpy::ndarray(npy_intp rows, npy_intp cols)
{
    return {rows, cols};
}

CNdArray CNumpy::rand(const npy_intp rows, const npy_intp cols)
{
    auto result = CNumpy::ndarray(rows, cols);
    auto *data = (float *) PyArray_DATA(result.ndarray);

    for (npy_intp i = 0; i < result.size; ++i)
    {
        data[i] = dist(gen);
    }
    return result;
}

CNdArray CNumpy::zeros(const npy_intp rows, const npy_intp cols)
{
    auto result = CNumpy::ndarray(rows, cols);
    auto *data = (float *) PyArray_DATA(result.ndarray);

    for (npy_intp i = 0; i < result.size; ++i)
    {
        data[i] = 0;
    }
    return result;
}

CNdArray::CNdArray(const npy_intp rows, const npy_intp cols) : dims{rows, cols}
{
    ndarray = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT);
    size = PyArray_SIZE(ndarray);
}

CNdArray::~CNdArray()
{
    if (ndarray)
    {
        Py_DECREF(ndarray);
    }
}

std::ostream &operator<<(std::ostream &os, const CNdArray &arr)
{
    PyObject *repr = PyObject_Repr((PyObject *) arr.ndarray);
    const char *str = PyUnicode_AsUTF8(repr);
    os << str << "\n";
    Py_DECREF(repr);
    return os;
}

float CNdArray::operator()(const int y, const int x) const
{
    assert(y < dims[0]);
    assert(x < dims[1]);
    const auto *value_ptr = (float *) PyArray_GETPTR2(ndarray, y, x);
    return *value_ptr;
}

float &CNdArray::operator()(const int y, const int x)
{
    assert(y < dims[0]);
    assert(x < dims[1]);
    auto *ptr = (float *) PyArray_GETPTR2(ndarray, y, x);
    return *ptr;
}

npy_intp CNdArray::rows() const
{
    return dims[0];
}

npy_intp CNdArray::cols() const
{
    return dims[1];
}

float CNumpy::max(const CNdArray &ndarray)
{
    const auto *data = (float *) PyArray_DATA(ndarray.ndarray);

    float max_val = -FLT_MAX;
    for (npy_intp i = 0; i < ndarray.size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

CNdArray &CNdArray::operator/=(const float div)
{
    if (div == 0.0)
    {
        throw std::invalid_argument("Division by zero.");
    }
    auto *data = (float *) PyArray_DATA(ndarray);
    for (npy_intp i = 0; i < size; ++i)
    {
        data[i] /= div;
    }
    return *this;
}

CNdArray CNdArray::operator-(const float sub) const
{
    auto result = CNdArray(dims[0], dims[1]);
    const auto *data = (float *) PyArray_DATA(ndarray);
    auto *result_data = (float *) PyArray_DATA(result.ndarray);
    for (npy_intp i = 0; i < size; ++i)
    {
        result_data[i] = data[i] - sub;
    }
    return result;
}

CNdArray CNdArray::transpose() const
{
    auto transposed = CNdArray(dims[1], dims[0]);
    transposed.ndarray = (PyArrayObject *) PyArray_Transpose(ndarray, NULL);
    return transposed;
}

CNdArray &CNdArray::operator=(const CNdArray &other)
{
    if (this != &other)
    {
        if (ndarray)
        {
            Py_DECREF(ndarray);
        }
        ndarray = (PyArrayObject *) PyArray_SimpleNew(2, other.dims, NPY_FLOAT);
        size = PyArray_SIZE(ndarray);

        auto *data = (float *) PyArray_DATA(ndarray);
        const auto *other_data = (float *) PyArray_DATA(other.ndarray);
        memcpy(data, other_data, size * sizeof(float));
    }
    return *this;
}

NeuralNet_CNumpy::NeuralNet_CNumpy(
        const int num_features, const int hidden_layer_size, const int categories)
{
    W1 = CNumpy::rand(hidden_layer_size, num_features) - 0.5f;
    b1 = CNumpy::rand(hidden_layer_size, 1) - 0.5f;
    W2 = CNumpy::rand(categories, hidden_layer_size) - 0.5f;
    b2 = CNumpy::rand(categories, 1) - 0.5f;
}
