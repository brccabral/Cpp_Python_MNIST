#pragma once

#include <vector>
#include <iostream>

extern "C"
{
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
}

class CNdArray
{
public:

    explicit CNdArray(PyArrayObject *arr);
    CNdArray(const CNdArray &other);
    CNdArray(CNdArray &&other) noexcept;
    CNdArray &operator=(const CNdArray &other);
    CNdArray &operator=(CNdArray &&other) noexcept;
    ~CNdArray();

    friend std::ostream &operator<<(std::ostream &os, const CNdArray &arr);

    double operator()(int y, int x) const;
    double &operator()(int y, int x);

    [[nodiscard]] npy_intp rows() const;
    [[nodiscard]] npy_intp cols() const;

    CNdArray &operator/=(double value);
    CNdArray operator/(long value) const;
    CNdArray operator-(double value) const;
    CNdArray operator-(const CNdArray &other) const;
    CNdArray operator*(const CNdArray &other) const;
    CNdArray operator*(double value) const;
    CNdArray operator+(const CNdArray &other) const;
    CNdArray operator/(const CNdArray &other) const;
    CNdArray operator==(const CNdArray &other) const;
    CNdArray operator>(const CNdArray &other) const;
    CNdArray operator>(double value) const;
    [[nodiscard]] CNdArray dot(const CNdArray &other) const;
    [[nodiscard]] CNdArray transpose() const;
    [[nodiscard]] CNdArray reshape(npy_intp d1, npy_intp d2) const;


private:

    friend class CNumpy;
    friend class NeuralNet_CNumpy;

    PyArrayObject *ndarray{};

    npy_intp *dims{};
    npy_intp size{};
    int ndim{};
};

class CNumpy
{
public:

    CNumpy(const CNumpy &other) = delete;
    CNumpy(CNumpy &&other) = delete;
    CNumpy &operator=(const CNumpy &other) = delete;
    CNumpy &operator=(CNumpy &&other) = delete;

    // singleton
    static CNumpy &instance()
    {
        static CNumpy instance;
        return instance;
    }

    PyObject *cnumpy{};

    PyObject *cnumpy_ndarray{};
    static CNdArray ndarray(npy_intp rows, npy_intp cols);
    PyObject *cnumpy_random{};
    PyObject *cnumpy_random_default_rng{};
    mutable PyObject *default_rng{};
    mutable PyObject *default_rng_random{};
    static CNdArray rng_random(npy_intp rows, npy_intp cols);
    static void random_seed(long seed);
    PyObject *cnumpy_zeros{};
    static CNdArray zeros(npy_intp rows, npy_intp cols);

    PyObject *cnumpy_add{};
    static CNdArray add(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_max{};
    [[nodiscard]] static double max(const CNdArray &a);
    PyObject *cnumpy_subtract{};
    static CNdArray subtract(const CNdArray &a, double value);
    static CNdArray subtract(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_dot{};
    static CNdArray dot(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_maximum{};
    static CNdArray maximum(const CNdArray &a, double value);
    PyObject *cnumpy_exp{};
    static CNdArray exp(const CNdArray &a);
    PyObject *cnumpy_sum{};
    static double sum(const CNdArray &a);
    static CNdArray sum(const CNdArray &a, long axis);
    PyObject *cnumpy_divide{};
    static CNdArray divide(const CNdArray &a, const CNdArray &b);
    static CNdArray divide(const CNdArray &a, long value);
    PyObject *cnumpy_argmax{};
    static CNdArray argmax(const CNdArray &a, long axis);
    PyObject *cnumpy_equal{};
    static CNdArray equal(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_greater{};
    static CNdArray greater(const CNdArray &a, const CNdArray &b);
    static CNdArray greater(const CNdArray &a, double value);
    PyObject *cnumpy_multiply{};
    static CNdArray multiply(const CNdArray &a, const CNdArray &b);
    static CNdArray multiply(const CNdArray &a, double value);

    PyObject *cnumpy_transpose{};
    static CNdArray transpose(const CNdArray &a);
    PyObject *cnumpy_reshape{};
    static CNdArray reshape(const CNdArray &a, npy_intp d1, npy_intp d2);

private:

    CNumpy();
    ~CNumpy();
    static CNdArray create_ndarray(npy_intp rows, npy_intp cols, PyObject *callable);

    void finalize() const;
};


class NeuralNet_CNumpy
{
public:

    NeuralNet_CNumpy(int num_features, int hidden_layer_size, int categories);

    CNdArray forward_prop(const CNdArray &X);
    void back_prop(const CNdArray &X, const CNdArray &target, float alpha);

    static CNdArray one_hot_encode(const CNdArray &Z);
    static CNdArray ReLU(const CNdArray &Z);
    static CNdArray deriv_ReLU(const CNdArray &Z);
    static CNdArray softmax(const CNdArray &Z);
    static CNdArray get_predictions(const CNdArray &A2);
    static double get_correct_prediction(const CNdArray &predictions, const CNdArray &Y);
    static double get_accuracy(double correct_prediction, npy_intp size);

    // layers
    CNdArray W1{nullptr}, b1{nullptr}, W2{nullptr}, b2{nullptr};
    // back prop
    CNdArray Z1{nullptr}, A1{nullptr}, Z2{nullptr}, A2{nullptr};
};
