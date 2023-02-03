#include <MNIST/MNIST_Dataset.hpp>
#include <fstream>
#include <stdexcept>

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

MNIST_Dataset::MNIST_Dataset(const char *image_filename,
                             const char *label_filename,
                             int image_magic,
                             int label_magic)
    : _image_filename(image_filename), _label_filename(label_filename), _image_magic(image_magic), _label_magic(label_magic){};

void MNIST_Dataset::save_dataset_as_png(std::string save_dir)
{
    for (MNIST_Image img : _images)
    {
        img.save_as_png(save_dir);
    }
};

void MNIST_Dataset::save_dataset_as_csv(std::string save_dir)
{
    for (MNIST_Image img : _images)
    {
        img.save_as_csv(save_dir);
    }
};

Eigen::MatrixXf MNIST_Dataset::to_matrix()
{
    int rows = _images.size();
    int cols = _images.at(0)._rows * _images.at(0)._cols + 1;

    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            mat(i, j) = (unsigned char)_images.at(i)._pixels[j];
        }
        mat(i, cols - 1) = float(_images.at(i)._label);
    }
    return mat;
};

void MNIST_Dataset::read_mnist_db(const int max_items)
{
    // Open files
    std::ifstream image_file(_image_filename, std::ios::in | std::ios::binary);
    if (!image_file.is_open())
        throw std::invalid_argument("Failed open image file.");

    std::ifstream label_file(_label_filename, std::ios::in | std::ios::binary);
    if (!label_file.is_open())
    {
        image_file.close();
        throw std::invalid_argument("Failed open label file.");
    }

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = swap_endian(magic);
    if (magic != _image_magic)
    {
        image_file.close();
        label_file.close();
        throw std::invalid_argument("Incorrect image file magic " + magic);
    }
    image_file.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));
    num_items = swap_endian(num_items);
    image_file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    cols = swap_endian(cols);

    label_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = swap_endian(magic);
    if (magic != _label_magic)
    {
        image_file.close();
        label_file.close();
        throw std::invalid_argument("Incorrect label file magic " + magic);
    }

    label_file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = swap_endian(num_labels);
    if (num_items != num_labels)
    {
        image_file.close();
        label_file.close();
        throw std::invalid_argument("image file nums should equal to label num");
    }

    int n_items = max_items;
    if (max_items > num_items)
    {
        n_items = num_items;
    }

    char label;
    char *pixels = new char[rows * cols];

    for (int item_id = 0; item_id < n_items; ++item_id)
    {
        // read image pixel
        image_file.read(pixels, rows * cols);
        // read label
        label_file.read(&label, 1);

        MNIST_Image m_image(rows, cols, int(label), pixels, item_id);

        _images.push_back(m_image);
    }

    delete[] pixels;
    image_file.close();
    label_file.close();
};