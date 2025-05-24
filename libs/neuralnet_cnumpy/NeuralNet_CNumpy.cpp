#include <NeuralNet_CNumpy/NeuralNet_CNumpy.h>
#include <stdexcept>
#include <climits>
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

    default_rng = PyObject_CallObject(cnumpy_random_default_rng, NULL);
    if (!default_rng)
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not call default_rng.");
    }

    default_rng_random = PyObject_GetAttrString(default_rng, "random");
    if (!default_rng_random || !PyCallable_Check(default_rng_random))
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

    cnumpy_max = PyObject_GetAttrString(cnumpy, "max");
    if (!cnumpy_max || !PyCallable_Check(cnumpy_max))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.max.");
    }

    cnumpy_subtract = PyObject_GetAttrString(cnumpy, "subtract");
    if (!cnumpy_subtract || !PyCallable_Check(cnumpy_subtract))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.subtract.");
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

    cnumpy_equal = PyObject_GetAttrString(cnumpy, "equal");
    if (!cnumpy_equal || !PyCallable_Check(cnumpy_equal))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.equal.");
    }

    cnumpy_greater = PyObject_GetAttrString(cnumpy, "greater");
    if (!cnumpy_greater || !PyCallable_Check(cnumpy_greater))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.greater.");
    }

    cnumpy_multiply = PyObject_GetAttrString(cnumpy, "multiply");
    if (!cnumpy_multiply || !PyCallable_Check(cnumpy_multiply))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.multiply.");
    }

    cnumpy_reshape = PyObject_GetAttrString(cnumpy, "reshape");
    if (!cnumpy_reshape || !PyCallable_Check(cnumpy_reshape))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.reshape.");
    }

    cnumpy_transpose = PyObject_GetAttrString(cnumpy, "transpose");
    if (!cnumpy_transpose || !PyCallable_Check(cnumpy_transpose))
    {
        PyErr_Print();
        finalize();
        throw std::invalid_argument("Could not get numpy.transpose.");
    }
}

CNumpy::~CNumpy()
{
    finalize();
}

void CNumpy::finalize() const
{
    Py_XDECREF(default_rng_random);
    Py_XDECREF(default_rng);
    Py_XDECREF(cnumpy_random_default_rng);
    Py_XDECREF(cnumpy_random);
    Py_XDECREF(cnumpy_ndarray);
    Py_XDECREF(cnumpy_zeros);
    Py_XDECREF(cnumpy_add);
    Py_XDECREF(cnumpy_max);
    Py_XDECREF(cnumpy_subtract);
    Py_XDECREF(cnumpy_dot);
    Py_XDECREF(cnumpy_maximum);
    Py_XDECREF(cnumpy_exp);
    Py_XDECREF(cnumpy_sum);
    Py_XDECREF(cnumpy_divide);
    Py_XDECREF(cnumpy_argmax);
    Py_XDECREF(cnumpy_equal);
    Py_XDECREF(cnumpy_greater);
    Py_XDECREF(cnumpy_reshape);
    Py_XDECREF(cnumpy_transpose);
    Py_DECREF(cnumpy);
    Py_Finalize();
}

CNdArray CNumpy::create_ndarray(const npy_intp rows, const npy_intp cols, PyObject *callable)
{
    PyObject *shape;
    if (cols > 0)
    {
        shape = PyTuple_Pack(2, PyLong_FromLong(rows), PyLong_FromLong(cols));
    }
    else
    {
        shape = PyTuple_Pack(1, PyLong_FromLong(rows));
    }
    PyObject *args = PyTuple_Pack(1, shape);

    const auto ndarray = (PyArrayObject *) PyObject_Call(callable, args, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR create_ndarray.");
    }

    Py_DECREF(args);
    Py_DECREF(shape);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::ndarray(const npy_intp rows, const npy_intp cols)
{
    return create_ndarray(rows, cols, np.cnumpy_ndarray);
}

CNdArray CNumpy::rng_random(const npy_intp rows, const npy_intp cols)
{
    return create_ndarray(rows, cols, np.default_rng_random);
}

void CNumpy::random_seed(const long seed)
{
    Py_XDECREF(np.default_rng);
    Py_XDECREF(np.default_rng_random);

    auto *s = PyLong_FromLong(seed);
    PyObject *args = PyTuple_Pack(1, s);

    np.default_rng = PyObject_CallObject(np.cnumpy_random_default_rng, args);
    if (!np.default_rng)
    {
        PyErr_Print();
        throw std::invalid_argument("ERROR CNumpy::random_seed default_rng.");
    }

    np.default_rng_random = PyObject_GetAttrString(np.default_rng, "random");
    if (!np.default_rng_random || !PyCallable_Check(np.default_rng_random))
    {
        PyErr_Print();
        throw std::invalid_argument("ERROR CNumpy::random_seed default_rng_random.");
    }

    Py_DECREF(s);
    Py_DECREF(args);
}

CNdArray CNumpy::zeros(const npy_intp rows, const npy_intp cols)
{
    return create_ndarray(rows, cols, np.cnumpy_zeros);
}

CNdArray CNumpy::add(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_add, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::add.");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::subtract(const CNdArray &a, const double sub)
{
    const auto *bo = PyFloat_FromDouble(sub);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_subtract, a.ndarray, bo, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::subtract(a,sub).");
    }

    Py_DECREF(bo);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::subtract(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_subtract, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::subtract(a,b).");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::dot(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_dot, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::dot.");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::maximum(const CNdArray &a, const double b)
{
    const auto *bo = PyFloat_FromDouble(b);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_maximum, a.ndarray, bo, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::maximum.");
    }

    Py_DECREF(bo);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::exp(const CNdArray &a)
{
    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_exp, a.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::exp.");
    }

    CNdArray result{ndarray};
    return result;
}

double CNumpy::sum(const CNdArray &a)
{
    auto *s = PyObject_CallFunctionObjArgs(np.cnumpy_sum, a.ndarray, NULL);
    if (!s)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::sum(a)->double.");
    }
    const double value = PyFloat_AsDouble(s);
    Py_DECREF(s);
    return value;
}

CNdArray CNumpy::sum(const CNdArray &a, const long axis)
{
    const auto *ax = PyLong_FromLong(axis);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_sum, a.ndarray, ax, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::sum(a,axis).");
    }

    Py_DECREF(ax);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::divide(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_divide, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::divide(a,b).");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::divide(const CNdArray &a, const long div)
{
    const auto *d = PyLong_FromLong(div);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_divide, a.ndarray, d, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::divide(a,div).");
    }

    Py_DECREF(d);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::argmax(const CNdArray &a, const long axis)
{
    const auto *ax = PyLong_FromLong(axis);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_argmax, a.ndarray, ax, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::argmax.");
    }

    Py_DECREF(ax);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::equal(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_equal, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::equal.");
    }

    CNdArray result{ndarray};
    return result;
}

double CNumpy::max(const CNdArray &a)
{
    auto *m = PyObject_CallFunctionObjArgs(np.cnumpy_max, a.ndarray, NULL);
    if (!m)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::max.");
    }
    const double value = PyFloat_AsDouble(m);
    Py_DECREF(m);
    return value;
}

CNdArray CNumpy::greater(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_greater, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::greater(a,b).");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::greater(const CNdArray &a, const double value)
{
    const auto *v = PyFloat_FromDouble(value);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_greater, a.ndarray, v, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::greater(a,value).");
    }

    Py_DECREF(v);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::multiply(const CNdArray &a, const CNdArray &b)
{
    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_multiply, a.ndarray, b.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::multiply.");
    }

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::multiply(const CNdArray &a, const double value)
{
    const auto *v = PyFloat_FromDouble(value);

    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_multiply, a.ndarray, v, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::multiply.");
    }

    Py_DECREF(v);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::reshape(const CNdArray &a, const npy_intp d1, const npy_intp d2)
{
    PyObject *shape = PyTuple_Pack(2, PyLong_FromLong(d1), PyLong_FromLong(d2));

    const auto ndarray = (PyArrayObject *) PyObject_CallFunctionObjArgs(
            np.cnumpy_reshape, a.ndarray, shape, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::reshape.");
    }

    Py_DECREF(shape);

    CNdArray result{ndarray};
    return result;
}

CNdArray CNumpy::transpose(const CNdArray &a)
{
    const auto ndarray =
            (PyArrayObject *) PyObject_CallFunctionObjArgs(np.cnumpy_transpose, a.ndarray, NULL);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR CNumpy::transpose.");
    }

    CNdArray result{ndarray};
    return result;
}


CNdArray::CNdArray(PyArrayObject *arr)
{
    if (arr && PyArray_Check(arr))
    {
        ndarray = arr;
        dims = PyArray_DIMS(ndarray);
        ndim = PyArray_NDIM(ndarray);
        size = PyArray_SIZE(ndarray);
    }
}

CNdArray::CNdArray(const CNdArray &other)
{
    PyObject *shape;
    if (other.ndim == 1)
    {
        shape = PyTuple_Pack(1, PyLong_FromLong(other.dims[0]));
    }
    else
    {
        shape = PyTuple_Pack(2, PyLong_FromLong(other.dims[0]), PyLong_FromLong(other.dims[1]));
    }
    PyObject *args = PyTuple_Pack(1, shape);

    ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);
    if (!ndarray)
    {
        PyErr_Print();
        throw std::runtime_error("ERROR copy ctor.");
    }

    Py_DECREF(args);
    Py_DECREF(shape);

    dims = PyArray_DIMS(ndarray);
    ndim = PyArray_NDIM(ndarray);
    size = PyArray_SIZE(ndarray);

    auto *data = (double *) PyArray_DATA(ndarray);
    const auto *other_data = (double *) PyArray_DATA(other.ndarray);
    std::memcpy(data, other_data, size * sizeof(double));
}

CNdArray::CNdArray(CNdArray &&other) noexcept
{
    ndarray = other.ndarray;
    if (ndarray && PyArray_Check(ndarray))
    {
        dims = PyArray_DIMS(ndarray);
        ndim = PyArray_NDIM(ndarray);
        size = PyArray_SIZE(ndarray);
    }
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
    if (ndim > 1)
    {
        return dims[1];
    }
    return 0;
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

CNdArray CNdArray::operator/(const long div) const
{
    if (div == 0)
    {
        throw std::invalid_argument("Division by zero.");
    }
    return CNumpy::divide(*this, div);
}

CNdArray CNdArray::operator-(const double sub) const
{
    return CNumpy::subtract(*this, sub);
}

CNdArray CNdArray::operator-(const CNdArray &other) const
{
    return CNumpy::subtract(*this, other);
}

CNdArray CNdArray::operator*(const CNdArray &other) const
{
    return CNumpy::multiply(*this, other);
}

CNdArray CNdArray::operator*(const double value) const
{
    return CNumpy::multiply(*this, value);
}

CNdArray CNdArray::operator+(const CNdArray &other) const
{
    return CNumpy::add(*this, other);
}

CNdArray CNdArray::operator/(const CNdArray &other) const
{
    return CNumpy::divide(*this, other);
}

CNdArray CNdArray::operator==(const CNdArray &other) const
{
    return CNumpy::equal(*this, other);
}

CNdArray CNdArray::operator>(const CNdArray &other) const
{
    return CNumpy::greater(*this, other);
}

CNdArray CNdArray::operator>(const double value) const
{
    return CNumpy::greater(*this, value);
}

CNdArray CNdArray::transpose() const
{
    return CNumpy::transpose(*this);
}

CNdArray CNdArray::dot(const CNdArray &other) const
{
    return CNumpy::dot(*this, other);
}

CNdArray &CNdArray::operator=(const CNdArray &other)
{
    if (this != &other)
    {
        if (ndarray)
        {
            Py_DECREF(ndarray);
        }

        PyObject *shape;
        if (other.ndim == 1)
        {
            shape = PyTuple_Pack(1, PyLong_FromLong(other.dims[0]));
        }
        else
        {
            shape = PyTuple_Pack(2, PyLong_FromLong(other.dims[0]), PyLong_FromLong(other.dims[1]));
        }
        PyObject *args = PyTuple_Pack(1, shape);

        ndarray = (PyArrayObject *) PyObject_CallObject(np.cnumpy_ndarray, args);
        if (!ndarray)
        {
            PyErr_Print();
            throw std::runtime_error("ERROR copy assign.");
        }

        Py_DECREF(args);
        Py_DECREF(shape);

        dims = PyArray_DIMS(ndarray);
        ndim = PyArray_NDIM(ndarray);
        size = PyArray_SIZE(ndarray);

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
        if (ndarray)
        {
            Py_DECREF(ndarray);
        }
        ndarray = other.ndarray;
        if (ndarray && PyArray_Check(ndarray))
        {
            dims = PyArray_DIMS(ndarray);
            ndim = PyArray_NDIM(ndarray);
            size = PyArray_SIZE(ndarray);
        }
        other.ndarray = nullptr;
    }
    return *this;
}

CNdArray CNdArray::reshape(const npy_intp d1, const npy_intp d2) const
{
    return CNumpy::reshape(*this, d1, d2);
}


NeuralNet_CNumpy::NeuralNet_CNumpy(
        const int num_features, const int hidden_layer_size, const int categories)
{
    W1 = CNumpy::rng_random(hidden_layer_size, num_features) - 0.5f;
    b1 = CNumpy::rng_random(hidden_layer_size, 1) - 0.5f;
    W2 = CNumpy::rng_random(categories, hidden_layer_size) - 0.5f;
    b2 = CNumpy::rng_random(categories, 1) - 0.5f;
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
    Z1 = W1.dot(X);
    Z1 = Z1 + b1;
    A1 = ReLU(Z1);

    Z2 = W2.dot(A1);
    Z2 = Z2 + b2;
    A2 = softmax(Z2);

    return A2;
}

void NeuralNet_CNumpy::back_prop(const CNdArray &X, const CNdArray &target, const float alpha)
{
    const auto y_size = target.cols();

    const auto dZ2 = (A2 - target) / y_size;
    const auto dW2 = dZ2.dot(A1.transpose());
    const auto db2 = CNumpy::sum(dZ2, 1).reshape(b2.rows(), b2.cols());

    const auto dZ1 = W2.transpose().dot(dZ2) * deriv_ReLU(Z1);
    const auto dW1 = dZ1.dot(X.transpose());
    const auto db1 = CNumpy::sum(dZ1, 1).reshape(b1.rows(), b1.cols());

    W1 = W1 - dW1 * alpha;
    b1 = b1 - db1 * alpha;
    W2 = W2 - dW2 * alpha;
    b2 = b2 - db2 * alpha;
}

CNdArray NeuralNet_CNumpy::ReLU(const CNdArray &Z)
{
    return CNumpy::maximum(Z, 0.0);
}

CNdArray NeuralNet_CNumpy::deriv_ReLU(const CNdArray &Z)
{
    return Z > 0.0;
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

double NeuralNet_CNumpy::get_correct_prediction(const CNdArray &predictions, const CNdArray &Y)
{
    return CNumpy::sum(predictions == Y);
}

double NeuralNet_CNumpy::get_accuracy(const double correct_prediction, const npy_intp size)
{
    return correct_prediction / size;
}
