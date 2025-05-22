#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>
#include <climits>
#include <cfloat>
#include <cstring>

const auto &np = CNumpy::instance();

bool init_numpy()
{
    import_array1(false);
    return true;
}

std::ostream &operator<<(std::ostream &os, const CNdArray &arr)
{
    PyObject *repr = PyObject_Repr((PyObject *) arr.ndarray);
    const char *str = PyUnicode_AsUTF8(repr);
    os << str << "\n";
    Py_DECREF(repr);
    return os;
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

    cnumpy_dot = PyObject_GetAttrString(cnumpy, "dot");
    if (!cnumpy_dot || !PyCallable_Check(cnumpy_dot))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.dot.");
    }

    cnumpy_maximum = PyObject_GetAttrString(cnumpy, "maximum");
    if (!cnumpy_maximum || !PyCallable_Check(cnumpy_maximum))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.maximum.");
    }

    cnumpy_exp = PyObject_GetAttrString(cnumpy, "exp");
    if (!cnumpy_exp || !PyCallable_Check(cnumpy_exp))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.exp.");
    }

    cnumpy_sum = PyObject_GetAttrString(cnumpy, "sum");
    if (!cnumpy_sum || !PyCallable_Check(cnumpy_sum))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.sum.");
    }

    cnumpy_divide = PyObject_GetAttrString(cnumpy, "divide");
    if (!cnumpy_divide || !PyCallable_Check(cnumpy_divide))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.divide.");
    }

    cnumpy_argmax = PyObject_GetAttrString(cnumpy, "argmax");
    if (!cnumpy_argmax || !PyCallable_Check(cnumpy_argmax))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.argmax.");
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
    Py_XDECREF(cnumpy_dot);
    Py_XDECREF(cnumpy_maximum);
    Py_XDECREF(cnumpy_exp);
    Py_XDECREF(cnumpy_sum);
    Py_XDECREF(cnumpy_divide);
    Py_XDECREF(cnumpy_argmax);
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
    PyObject *args = PyTuple_Pack(1, shape);

    result.ndarray = (PyArrayObject *) PyObject_Call(np.rng_random, args, NULL);

    Py_DECREF(args);
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

CNdArray CNumpy::dot(const CNdArray &a, const CNdArray &b)
{
    auto result = CNdArray(a.rows(), b.cols());
    result.ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_dot, a.ndarray, b.ndarray, NULL);
    return result;
}

CNdArray CNumpy::maximum(const CNdArray &a, const double b)
{
    const auto bo = PyFloat_FromDouble(b);
    auto result = CNdArray(a.rows(), a.cols());
    result.ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_maximum, a.ndarray, bo, NULL);
    // TODO : verify the returned dimensions
    Py_DECREF(bo);
    return result;
}

CNdArray CNumpy::exp(const CNdArray &a)
{
    auto result = CNdArray(a.rows(), a.cols());
    result.ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_exp, a.ndarray, NULL);
    return result;
}

CNdArray CNumpy::sum(const CNdArray &a, const long axis)
{
    auto result = CNdArray(a.rows(), a.cols());
    const auto ax = PyLong_FromLong(axis);
    result.ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_sum, a.ndarray, ax, NULL);
    Py_DECREF(ax);
    return result;
}

CNdArray CNumpy::divide(const CNdArray &a, const CNdArray &b)
{
    auto result = CNdArray(a.rows(), a.cols());
    result.ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_divide, a.ndarray, b.ndarray, NULL);
    // TODO : verify the returned dimensions
    return result;
}

CNdArray CNumpy::argmax(const CNdArray &a, const long axis)
{
    auto result = CNdArray(a.rows(), a.cols());
    const auto ax = PyLong_FromLong(axis);
    result.ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_argmax, a.ndarray, ax, NULL);
    Py_DECREF(ax);
    return result;
}

double CNumpy::max(const CNdArray &ndarray)
{
    const auto *data = (double *) PyArray_DATA(ndarray.ndarray);

    double max_val = -DBL_MAX;
    for (npy_intp i = 0; i < ndarray.size; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

CNdArray::CNdArray(const npy_intp rows, const npy_intp cols) : dims{rows, cols}
{
    size = rows * cols;
}

CNdArray::CNdArray(const CNdArray &other)
{
    std::memcpy(dims, other.dims, 2 * sizeof(npy_intp));
    PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(dims[0]), PyLong_FromLong(dims[1]));
    PyObject *args = PyTuple_Pack(1, shape);

    ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);

    Py_DECREF(args);
    Py_DECREF(shape);

    size = other.size;

    auto *data = (double *) PyArray_DATA(ndarray);
    const auto *other_data = (double *) PyArray_DATA(other.ndarray);
    std::memcpy(data, other_data, size * sizeof(double));
}

CNdArray::CNdArray(CNdArray &&other) noexcept
{
    std::memcpy(dims, other.dims, 2 * sizeof(npy_intp));
    size = other.size;
    ndarray = other.ndarray;
    other.ndarray = nullptr;
}

CNdArray::~CNdArray()
{
    Py_XDECREF(ndarray);
}

double CNdArray::operator()(const int y, const int x) const
{
    assert(y < dims[0]);
    assert(x < dims[1]);
    const auto *value_ptr = (double *) PyArray_GETPTR2(ndarray, y, x);
    return *value_ptr;
}

double &CNdArray::operator()(const int y, const int x)
{
    assert(y < dims[0]);
    assert(x < dims[1]);
    auto *ptr = (double *) PyArray_GETPTR2(ndarray, y, x);
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

CNdArray &CNdArray::operator/=(const double div)
{
    if (div == 0.0)
    {
        throw std::invalid_argument("Division by zero.");
    }
    auto *data = (double *) PyArray_DATA(ndarray);
    for (npy_intp i = 0; i < size; ++i)
    {
        data[i] /= div;
    }
    return *this;
}

CNdArray CNdArray::operator-(const double sub) const
{
    auto result = CNumpy::ndarray(dims[0], dims[1]);

    const auto *data = (double *) PyArray_DATA(ndarray);
    auto *result_data = (double *) PyArray_DATA(result.ndarray);
    for (npy_intp i = 0; i < size; ++i)
    {
        result_data[i] = data[i] - sub;
    }
    return result;
}

CNdArray CNdArray::operator*(const CNdArray &other) const
{
    return CNumpy::dot(*this, other);
}

CNdArray CNdArray::operator+(const CNdArray &other) const
{
    return CNumpy::add(*this, other);
}

CNdArray CNdArray::operator/(const CNdArray &other) const
{
    return CNumpy::divide(*this, other);
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
        std::memcpy(dims, other.dims, 2 * sizeof(npy_intp));

        PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(dims[0]), PyLong_FromLong(dims[1]));
        PyObject *args = PyTuple_Pack(1, shape);

        ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);

        Py_DECREF(args);
        Py_DECREF(shape);

        size = other.size;

        auto *data = (double *) PyArray_DATA(ndarray);
        const auto *other_data = (double *) PyArray_DATA(other.ndarray);
        std::memcpy(data, other_data, size * sizeof(double));
    }
    return *this;
}

CNdArray &CNdArray::operator=(CNdArray &&other) noexcept
{
    if (this != &other)
    {
        std::memcpy(dims, other.dims, 2 * sizeof(npy_intp));
        size = other.size;
        ndarray = other.ndarray;
        other.ndarray = nullptr;
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
    A1 = ReLU(Z1);

    Z2 = W2 * A1;
    Z2 = Z2 + b2;
    A2 = softmax(Z2);

    return A2;
}

CNdArray NeuralNet_CNumpy::ReLU(const CNdArray &Z)
{
    return CNumpy::maximum(Z, 0.0);
}

CNdArray NeuralNet_CNumpy::softmax(const CNdArray &Z)
{
    const auto e = CNumpy::exp(Z);
    const auto s = CNumpy::sum(e, 0);
    return e / s;
}

CNdArray NeuralNet_CNumpy::get_predictions(const CNdArray &A2)
{
    return CNumpy::argmax(A2, 0);
}
