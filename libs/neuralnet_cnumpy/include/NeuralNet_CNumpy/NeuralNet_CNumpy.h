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

    CNdArray() : ndarray(nullptr), dims{}, size() {};

    friend std::ostream &operator<<(std::ostream &os, const CNdArray &arr);

    float operator()(int y, int x) const;
    float &operator()(int y, int x);
    CNdArray &operator/=(float div);
    [[nodiscard]] CNdArray transpose() const;

    ~CNdArray();

    [[nodiscard]] npy_intp rows() const;
    [[nodiscard]] npy_intp cols() const;

private:

    friend class CNumpy;

    CNdArray(npy_intp rows, npy_intp cols);

    PyArrayObject *ndarray;

    npy_intp const dims[2];
    npy_intp size;
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

    static CNdArray ndarray(npy_intp rows, npy_intp cols);
    [[nodiscard]] static float max(const CNdArray &ndarray);

private:

    CNumpy();
    ~CNumpy();
};
