#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>

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

CNdArray CNumpy::ndarray(int nd, npy_intp const *dims, int ndtype)
{
    return {nd, dims, ndtype};
}

CNdArray::CNdArray(const int nd, npy_intp const *dims, const int ndtype)
    : nd(nd), dims(dims), ndtype(ndtype)
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
