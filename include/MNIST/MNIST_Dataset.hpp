#pragma once

#include <vector>
#include "MNIST_Image.hpp"

class MNIST_Dataset
{
private:

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

    void read_mnist_db(const int max_items);
    size_t get_images_length() const;
    int get_label_from_index(int index) const;

    std::vector<MNIST_Image> _images;
};
