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

    friend std::ostream &operator<<(std::ostream &os, const CNdArray &arr);

    float operator()(int y, int x) const;
    float &operator()(int y, int x);
    CNdArray &operator/=(float div);
    CNdArray transpose() const;

    ~CNdArray();

    [[nodiscard]] npy_intp rows() const;
    [[nodiscard]] npy_intp cols() const;

    int ndtype;

private:

    friend class CNumpy;

    CNdArray(int nd, npy_intp const dims[2], int ndtype);

    PyArrayObject *ndarray;

    int nd;
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

    CNdArray ndarray(int nd, npy_intp const *dims, int ndtype) const;
    float max(const CNdArray &ndarray) const;

private:

    CNumpy();
    ~CNumpy();
};
