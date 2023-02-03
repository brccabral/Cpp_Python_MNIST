#pragma once

#include <iostream>
#include <vector>
#include "MNIST_Image.hpp"
#include <eigen3/Eigen/Dense>

class MNIST_Dataset
{
private:
    std::vector<MNIST_Image> _images;
    const char *_image_filename;
    const char *_label_filename;
    int _image_magic;
    int _label_magic;

public:
    MNIST_Dataset(const char *image_filename,
                  const char *label_filename,
                  int image_magic,
                  int label_magic);

    void save_dataset_as_png(std::string save_dir);

    void save_dataset_as_csv(std::string save_dir);

    Eigen::MatrixXf to_matrix();

    void read_mnist_db(const int max_items);
};