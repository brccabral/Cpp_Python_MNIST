#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

template <typename T>
std::string to_string(const T &value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

class MNIST_Image
{
public:
    uint32_t _rows;
    uint32_t _cols;
    int _label;
    char *_pixels;
    int _db_item_id;

public:
    MNIST_Image(uint32_t rows, uint32_t cols, int label, char *pixels, int item_id) : _rows(rows), _cols(cols), _label(label), _db_item_id(item_id)
    {
        _pixels = new char[_rows * _cols];
        memcpy(_pixels, pixels, _rows * _cols);
    }
    ~MNIST_Image()
    {
        free(_pixels);
    }
    void save_as_png(std::string save_dir)
    {
        cv::Mat image_tmp(_rows, _cols, CV_8UC1, _pixels);
        std::string filename = save_dir + "/" + std::to_string(_db_item_id) + "_" + std::to_string(_label) + ".png";
        cv::imwrite(filename, image_tmp);
    }
    void save_as_csv(std::string save_dir, bool append)
    {
        std::ofstream outfile;
        if (append)
            outfile.open(save_dir + "/res.txt", std::ios_base::app);
        else
            outfile.open(save_dir + "/res.txt");

        outfile << _label;
        for (int p = 0; p < _rows * _cols; p++)
        {
            outfile << ',' << std::to_string((unsigned char)_pixels[p]);
        }
        outfile << std::endl;
    }
};
std::ostream &operator<<(std::ostream &outs, const MNIST_Image &m)
{
    outs << m._label;
    for (int p = 0; p < m._rows * m._cols; p++)
    {
        outs << ',' << std::to_string((unsigned char)m._pixels[p]);
    }
    return outs;
};

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void read_mnist_cv(const char *image_filename, const char *label_filename, const int max_items, const char *save_dir)
{
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    if (!image_file.is_open())
    {
        std::cout << "Failed open image file. " << std::endl;
        return;
    }
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    if (!label_file.is_open())
    {
        std::cout << "Failed open label file. " << std::endl;
        return;
    }

    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = swap_endian(magic);
    if (magic != 2051)
    {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }

    image_file.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));
    num_items = swap_endian(num_items);
    image_file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    cols = swap_endian(cols);

    label_file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = swap_endian(magic);
    if (magic != 2049)
    {
        std::cout << "Incorrect image file magic: " << magic << std::endl;
        return;
    }
    label_file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = swap_endian(num_labels);
    if (num_items != num_labels)
    {
        std::cout << "image file nums should equal to label num" << std::endl;
        return;
    }

    int n_items = max_items;
    if (max_items > num_items)
    {
        n_items = num_items;
    }

    std::cout << "image and label num is: " << num_items << std::endl;
    std::cout << "image rows: " << rows << ", cols: " << cols << std::endl;

    char label;
    char *pixels = new char[rows * cols];

    for (int item_id = 0; item_id < n_items; ++item_id)
    {
        // read image pixel
        image_file.read(pixels, rows * cols);
        // read label
        label_file.read(&label, 1);

        MNIST_Image m_image(rows, cols, int(label), pixels, item_id);

        std::string sLabel = std::to_string(int(label));
        std::cout << "lable is: " << sLabel << std::endl;

        m_image.save_as_png(save_dir);
        if (item_id == 0)
            m_image.save_as_csv(save_dir, false);
        else
            m_image.save_as_csv(save_dir, true);
    }

    delete[] pixels;
    image_file.close();
    label_file.close();
}

int main()
{
    std::string base_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST";
    std::string save_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST/train";
    std::string img_path = base_dir + "/train-images.idx3-ubyte";
    std::string label_path = base_dir + "/train-labels.idx1-ubyte";
    const int max_items = 10;

    read_mnist_cv(img_path.c_str(), label_path.c_str(), max_items, save_dir.c_str());
}