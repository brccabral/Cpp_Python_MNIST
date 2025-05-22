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

    CNdArray() = default;
    CNdArray(const CNdArray &other);
    CNdArray(CNdArray &&other) noexcept;
    friend std::ostream &operator<<(std::ostream &os, const CNdArray &arr);

    double operator()(int y, int x) const;
    double &operator()(int y, int x);
    CNdArray &operator/=(double div);
    CNdArray operator-(double sub) const;
    CNdArray operator*(const CNdArray &other) const;
    CNdArray operator+(const CNdArray &other) const;
    CNdArray operator/(const CNdArray &other) const;
    CNdArray operator==(const CNdArray &other) const;
    [[nodiscard]] CNdArray transpose() const;

    ~CNdArray();

    CNdArray &operator=(const CNdArray &other);
    CNdArray &operator=(CNdArray &&other) noexcept;

    [[nodiscard]] npy_intp rows() const;
    [[nodiscard]] npy_intp cols() const;

private:

    friend class CNumpy;
    friend class NeuralNet_CNumpy;

    CNdArray(npy_intp rows, npy_intp cols);

    PyArrayObject *ndarray{};

    npy_intp dims[2]{};
    npy_intp size{};
};

class CNumpy
{
public:

    CNumpy(const CNumpy &other) = delete;
    CNumpy(CNumpy &&other) = delete;
    CNumpy &operator=(const CNumpy &other) = delete;

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
    PyObject *rng{};
    PyObject *rng_random{};
    static CNdArray rand(npy_intp rows, npy_intp cols);
    PyObject *cnumpy_zeros{};
    static CNdArray zeros(npy_intp rows, npy_intp cols);
    [[nodiscard]] static double max(const CNdArray &ndarray);

    PyObject *cnumpy_add{};
    static CNdArray add(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_dot{};
    static CNdArray dot(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_maximum{};
    static CNdArray maximum(const CNdArray &a, double b);
    PyObject *cnumpy_exp{};
    static CNdArray exp(const CNdArray &a);
    PyObject *cnumpy_sum{};
    static double sum(const CNdArray &a);
    static CNdArray sum(const CNdArray &a, long axis);
    PyObject *cnumpy_divide{};
    static CNdArray divide(const CNdArray &a, const CNdArray &b);
    PyObject *cnumpy_argmax{};
    static CNdArray argmax(const CNdArray &a, long axis);
    PyObject *cnumpy_equal{};
    static CNdArray equal(const CNdArray &a, const CNdArray &b);

private:

    CNumpy();
    ~CNumpy();

    void finalize() const;
};


class NeuralNet_CNumpy
{
public:

    NeuralNet_CNumpy(int num_features, int hidden_layer_size, int categories);

    static CNdArray one_hot_encode(const CNdArray &Z);
    CNdArray forward_prop(const CNdArray &X);
    static CNdArray ReLU(const CNdArray &Z);
    static CNdArray softmax(const CNdArray &Z);
    static CNdArray get_predictions(const CNdArray &A2);

    // layers
    CNdArray W1{}, b1{}, W2{}, b2{};
    // back prop
    CNdArray Z1{}, A1{}, Z2{}, A2{};
    // gradients
    CNdArray dW1{}, dB1{}, dW2{}, dB2{};
};
