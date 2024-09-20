#pragma once

#include <iostream>
#include <stdint.h>

class MNIST_Image
{
public:

    uint32_t _rows;
    uint32_t _cols;
    int _label;
    char *_pixels;
    int _db_item_id;

public:

    MNIST_Image(uint32_t rows, uint32_t cols, int label, const char *pixels, int item_id);
    MNIST_Image(const MNIST_Image &other);
    ~MNIST_Image();

    void save_as_png(const std::string &save_dir) const;
    void save_as_csv(const std::string &save_filename) const;
};
