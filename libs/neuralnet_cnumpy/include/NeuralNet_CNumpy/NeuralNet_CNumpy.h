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

    ~CNdArray();

private:

    friend class CNumpy;

    CNdArray(int nd, npy_intp const *dims, int ndtype);

    PyArrayObject *ndarray;

    int nd;
    npy_intp const *dims;
    int ndtype;
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

    CNdArray ndarray(int nd, npy_intp const *dims, int ndtype);

private:

    CNumpy();
    ~CNumpy();
};
