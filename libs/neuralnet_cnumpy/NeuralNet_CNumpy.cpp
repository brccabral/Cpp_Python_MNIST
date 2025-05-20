#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>
#include <climits>
#include <cfloat>

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

CNdArray CNdArray::transpose() const
{
    auto transposed = CNdArray(dims[1], dims[0]);
    transposed.ndarray = (PyArrayObject *) PyArray_Transpose(ndarray, NULL);
    return transposed;
}
