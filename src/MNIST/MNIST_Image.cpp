#include <MNIST/MNIST_Image.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

std::ostream &operator<<(std::ostream &outs, const MNIST_Image &m)
{
    outs << m._label;
    for (int p = 0; p < m._rows * m._cols; p++)
    {
        outs << ',' << std::to_string((unsigned char) m._pixels[p]);
    }
    return outs;
}

MNIST_Image::MNIST_Image(const uint32_t rows, const uint32_t cols, const int label, const char *pixels,
        const int item_id)
    : _rows(rows), _cols(cols), _label(label), _db_item_id(item_id)
{
    _pixels = new char[_rows * _cols];
    memcpy(_pixels, pixels, rows * cols * sizeof(char));
}

MNIST_Image::MNIST_Image(const MNIST_Image &other)
    : _rows(other._rows), _cols(other._cols), _label(other._label), _db_item_id(other._db_item_id)
{
    _pixels = new char[_rows * _cols];
    memcpy(_pixels, other._pixels, other._rows * other._cols * sizeof(char));
}

MNIST_Image::~MNIST_Image()
{
    delete[] _pixels;
}

void MNIST_Image::save_as_png(const std::string &save_dir) const
{
    const cv::Mat image_tmp(_rows, _cols, CV_8UC1, _pixels);
    const std::string filename =
            save_dir + "/" + std::to_string(_db_item_id) + "_" + std::to_string(_label) + ".png";
    cv::imwrite(filename, image_tmp);
}

void MNIST_Image::save_as_csv(const std::string &save_filename) const
{
    std::ofstream outfile;
    if (_db_item_id == 0)
        outfile.open(save_filename);
    else
        outfile.open(save_filename, std::ios_base::app);

    outfile << _label;
    for (int p = 0; p < _rows * _cols; p++)
    {
        outfile << ',' << std::to_string((unsigned char) _pixels[p]);
    }
    outfile << "\n";
    outfile.close();
}
