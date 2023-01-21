#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

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

    std::ofstream result_file;
    std::string result_filename = "/res.txt";
    result_file.open(save_dir + result_filename);
    if (!result_file.is_open())
    {
        std::cout << "Failed open result file. " << std::endl;
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

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(1);

    std::string slash = "/";

    int one_hot[10] = {0};

    for (int item_id = 0; item_id < n_items; ++item_id)
    {
        // read image pixel
        image_file.read(pixels, rows * cols);
        // read label
        label_file.read(&label, 1);

        one_hot[int(label)] = 1;
        for (int encode = 0; encode < 10; encode++)
        {
            result_file << one_hot[encode] << " ";
        }
        result_file << "** ";
        one_hot[int(label)] = 0;

        std::string sLabel = std::to_string(int(label));
        std::cout << "lable is: " << sLabel << std::endl;
        // convert it to cv Mat, and show it
        cv::Mat image_tmp(rows, cols, CV_8UC1, pixels);
        // resize bigger for showing
        // cv::resize(image_tmp, image_tmp, cv::Size(100, 100));
        // cv::imshow(sLabel, image_tmp);
        std::string filename = save_dir + slash + std::to_string(item_id) + "_" + sLabel + ".png";
        cv::imwrite(filename, image_tmp, compression_params);
        // cv::waitKey(0);

        for (int p = 0; p < rows * cols; p++)
        {
            result_file << std::to_string((unsigned char)pixels[p]) << " ";
        }
        result_file << std::endl;
    }

    delete[] pixels;
    image_file.close();
    label_file.close();
    result_file.close();
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