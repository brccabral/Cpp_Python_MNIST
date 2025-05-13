#pragma once

#include <vector>
#include "MNIST_Image.hpp"

class MNIST_Dataset
{
private:

    const char *_image_filename;
    const char *_label_filename;
    uint _image_magic;
    uint _label_magic;

public:

    MNIST_Dataset(
            const char *image_filename, const char *label_filename, uint image_magic,
            uint label_magic);

#ifdef CV_SAVE_IMAGES
    void save_dataset_as_png(const std::string &save_dir);
    void save_dataset_as_csv(const std::string &save_filename);
#endif // CV_SAVE_IMAGES

    void read_mnist_db(const uint max_items);
    size_t get_images_length() const;
    int get_label_from_index(int index) const;

    std::vector<MNIST_Image> _images;
};
