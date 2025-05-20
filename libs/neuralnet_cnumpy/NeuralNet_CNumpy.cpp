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

CNdArray CNumpy::ndarray(int nd, npy_intp const *dims, int ndtype) const
{
    return {nd, dims, ndtype};
}

CNdArray::CNdArray(const int nd, npy_intp const dims[2], const int ndtype)
    : ndtype(ndtype), nd(nd), dims{dims[0], dims[1]}
{
    ndarray = (PyArrayObject *) PyArray_SimpleNew(nd, dims, ndtype);
}

CNdArray::~CNdArray()
{
    Py_DECREF(ndarray);
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

float CNumpy::max(const CNdArray &ndarray) const
{
    const auto *data = (float *) PyArray_DATA(ndarray.ndarray);
    const npy_intp size = PyArray_SIZE(ndarray.ndarray);

    float max_val = -FLT_MAX;
    for (npy_intp i = 0; i < size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}
