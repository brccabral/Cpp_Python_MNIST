#pragma once

#include <vector>
#include "MNIST_Image.hpp"
#include <Eigen/Dense>
#include <NumCpp.hpp>
#include <xtensor/xarray.hpp>

class MNIST_Dataset
{
private:

    std::vector<MNIST_Image> _images;
    const char *_image_filename;
    const char *_label_filename;
    int _image_magic;
    int _label_magic;

public:

    MNIST_Dataset(
            const char *image_filename, const char *label_filename, int image_magic,
            int label_magic);

    void save_dataset_as_png(const std::string &save_dir);
    void save_dataset_as_csv(const std::string &save_filename);

    [[nodiscard]] Eigen::MatrixXf to_matrix() const;
    [[nodiscard]] nc::NdArray<float> to_numcpp() const;
    [[nodiscard]] xt::xarray<float> to_xtensor() const;
    static Eigen::MatrixXf get_X(Eigen::MatrixXf &mat);
    static nc::NdArray<float> get_X(const nc::NdArray<float> &mat);
    static Eigen::VectorXf get_Y(Eigen::MatrixXf &mat);
    static nc::NdArray<float> get_Y(const nc::NdArray<float> &mat);

    void read_mnist_db(const int max_items);
    size_t get_images_length() const;
    int get_label_from_index(int index) const;
};
