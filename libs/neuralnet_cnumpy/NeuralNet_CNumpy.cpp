#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>
#include <climits>
#include <cfloat>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0.0, 1.0);

const auto &np = CNumpy::instance();

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

    cnumpy = PyImport_ImportModule("numpy");
    if (!cnumpy)
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not init numpy.");
    }

    cnumpy_ndarray = PyObject_GetAttrString(cnumpy, "ndarray");
    if (!cnumpy_ndarray || !PyCallable_Check(cnumpy_ndarray))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.ndarray.");
    }

    cnumpy_random = PyImport_ImportModule("numpy.random");
    if (!cnumpy_random)
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.random.");
    }

    cnumpy_random_default_rng = PyObject_GetAttrString(cnumpy_random, "default_rng");
    if (!cnumpy_random_default_rng || !PyCallable_Check(cnumpy_random_default_rng))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.random.default_rng.");
    }

    rng = PyObject_CallObject(cnumpy_random_default_rng, NULL);
    if (!rng)
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not call default_rng.");
    }

    rng_random = PyObject_GetAttrString(rng, "random");
    if (!rng_random || !PyCallable_Check(rng_random))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get default_rng.random.");
    }

    cnumpy_zeros = PyObject_GetAttrString(cnumpy, "zeros");
    if (!cnumpy_zeros || !PyCallable_Check(cnumpy_zeros))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.zeros.");
    }

    cnumpy_add = PyObject_GetAttrString(cnumpy, "add");
    if (!cnumpy_add || !PyCallable_Check(cnumpy_add))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.add.");
    }
}

CNumpy::~CNumpy()
{
    finalize();
}

void CNumpy::finalize() const
{
    Py_XDECREF(rng_random);
    Py_XDECREF(rng);
    Py_XDECREF(cnumpy_random_default_rng);
    Py_XDECREF(cnumpy_random);
    Py_XDECREF(cnumpy_ndarray);
    Py_XDECREF(cnumpy_zeros);
    Py_XDECREF(cnumpy_add);
    Py_DECREF(cnumpy);
    Py_Finalize();
}


CNdArray CNumpy::ndarray(const npy_intp rows, const npy_intp cols)
{
    auto result = CNdArray(rows, cols);
    PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
    PyObject *args = PyTuple_Pack(1, shape);

    result.ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);

    Py_DECREF(args);
    Py_DECREF(shape);
    return result;
}

CNdArray CNumpy::rand(const npy_intp rows, const npy_intp cols)
{
    auto result = CNdArray(rows, cols);
    PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
    PyObject *kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "size", shape);

    // result.ndarray = (PyArrayObject *) PyObject_CallMethod(np.rng, "random", NULL, NULL, kwargs);
    result.ndarray = (PyArrayObject *) PyObject_Call(np.rng_random, PyTuple_New(0), kwargs);
    if (!result.ndarray)
    {
        PyErr_Print();
    }

    Py_DECREF(kwargs);
    Py_DECREF(shape);
    return result;
}

CNdArray CNumpy::zeros(const npy_intp rows, const npy_intp cols)
{
    auto result = CNdArray(rows, cols);

    PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
    PyObject *args = PyTuple_Pack(1, shape);

    result.ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_zeros, args);

    Py_DECREF(args);
    Py_DECREF(shape);
    return result;
}

CNdArray CNumpy::add(const CNdArray &a, const CNdArray &b)
{
    auto result = CNdArray(a.rows(), a.cols());

    result.ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_add, a.ndarray, b.ndarray, NULL);
    return result;
}

CNdArray::CNdArray(const npy_intp rows, const npy_intp cols) : dims{rows, cols}
{
    size = rows * cols;
}

CNdArray::~CNdArray()
{
    Py_XDECREF(ndarray);
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
    auto result = CNumpy::ndarray(dims[0], dims[1]);

    const auto *data = (float *) PyArray_DATA(ndarray);
    auto *result_data = (float *) PyArray_DATA(result.ndarray);
    for (npy_intp i = 0; i < size; ++i)
    {
        result_data[i] = data[i] - sub;
    }
    return result;
}

CNdArray CNdArray::operator*(const CNdArray &mul) const
{
    assert(dims[1] == mul.dims[0]);
    auto result = CNdArray(dims[0], mul.dims[1]);
    result.ndarray =
            (PyArrayObject *) PyArray_MatrixProduct((PyObject *) ndarray, (PyObject *) mul.ndarray);
    return result;
}

CNdArray CNdArray::operator+(const CNdArray &add) const
{
    return CNumpy::add(*this, add);
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
        memcpy((void *) dims, other.dims, 2 * sizeof(npy_intp));

        PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(dims[0]), PyLong_FromLong(dims[1]));
        PyObject *args = PyTuple_Pack(1, shape);

        ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);

        Py_DECREF(args);
        Py_DECREF(shape);

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
    std::cout << W1 << std::endl;
    b1 = CNumpy::rand(hidden_layer_size, 1) - 0.5f;
    W2 = CNumpy::rand(categories, hidden_layer_size) - 0.5f;
    b2 = CNumpy::rand(categories, 1) - 0.5f;
}

CNdArray NeuralNet_CNumpy::one_hot_encode(const CNdArray &Z)
{
    auto o = CNumpy::zeros(Z.rows(), CNumpy::max(Z) + 1);

    for (int r = 0; r < Z.rows() - 1; r++)
    {
        o(r, int(Z(r, 0))) = 1;
    }
    return o.transpose();
}

CNdArray NeuralNet_CNumpy::forward_prop(const CNdArray &X)
{
    Z1 = W1 * X;
    Z1 = Z1 + b1;

    return Z1;
}
