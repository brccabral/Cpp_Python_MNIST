#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#ifdef DEBUG
#define DUMP_VAR(x) std::cout << #x "\n" \
                              << x << std::endl
#else
#define DUMP_VAR(x)
#endif

float get_float_matrix(Eigen::MatrixXf &M, int r, int c)
{
    return M(r, c);
}

float get_float_vector(Eigen::VectorXf &V, int r)
{
    return V(r);
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
        _pixels = new char[rows * cols];
        for (int i = 0; i < rows * cols; i++)
        {
            _pixels[i] = pixels[i];
        }
    }
    ~MNIST_Image()
    {
        // free(_pixels);
    }
    void save_as_png(std::string save_dir)
    {
        cv::Mat image_tmp(_rows, _cols, CV_8UC1, _pixels);
        std::string filename = save_dir + "/" + std::to_string(_db_item_id) + "_" + std::to_string(_label) + ".png";
        cv::imwrite(filename, image_tmp);
    }
    void save_as_csv(std::string save_dir)
    {
        std::ofstream outfile;
        if (_db_item_id == 0)
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

Eigen::MatrixXf to_matrix(std::vector<MNIST_Image> *dataset)
{
    int rows = dataset->size();
    int cols = dataset->at(0)._rows * dataset->at(0)._cols + 1;

    DUMP_VAR(rows);
    DUMP_VAR(cols);

    Eigen::MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            mat(i, j) = (unsigned char)dataset->at(i)._pixels[j];
        }
        mat(i, cols - 1) = float(dataset->at(i)._label);
    }
    return mat;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

std::vector<MNIST_Image> read_mnist_db(const char *image_filename, const char *label_filename, const int max_items, const char *save_dir, bool save_img)
{
    std::vector<MNIST_Image> dataset;

    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    if (!image_file.is_open())
    {
        std::cout << "Failed open image file. " << std::endl;
        return dataset;
    }
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    if (!label_file.is_open())
    {
        std::cout << "Failed open label file. " << std::endl;
        return dataset;
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
        return dataset;
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
        return dataset;
    }
    label_file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
    num_labels = swap_endian(num_labels);
    if (num_items != num_labels)
    {
        std::cout << "image file nums should equal to label num" << std::endl;
        return dataset;
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

        if (save_img)
            m_image.save_as_png(save_dir);

        m_image.save_as_csv(save_dir);

        dataset.push_back(m_image);
    }

    delete[] pixels;
    image_file.close();
    label_file.close();
    return dataset;
}

Eigen::MatrixXf ReLU(Eigen::MatrixXf &Z)
{
    return Z.cwiseMax(0);
}

Eigen::MatrixXf deriv_ReLU(Eigen::MatrixXf &Z)
{
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> b2 = (Z.array() > 0);
    return b2.unaryExpr([](const bool x)
                        { return x ? 1.0f : 0.0f; });
}

Eigen::MatrixXf Softmax(Eigen::MatrixXf &Z)
{
    Eigen::MatrixXf e = Z.array().exp();
    Eigen::MatrixXf s = e.colwise().sum();
    for (int c = 0; c < e.cols(); c++)
    {
        e.col(c) = e.col(c) / s(c);
    }
    return e;
}

void forward_prop(Eigen::MatrixXf &W1,
                  Eigen::VectorXf &b1,
                  Eigen::MatrixXf &W2,
                  Eigen::VectorXf &b2,
                  Eigen::MatrixXf &X,
                  Eigen::MatrixXf &Z1,
                  Eigen::MatrixXf &A1,
                  Eigen::MatrixXf &Z2,
                  Eigen::MatrixXf &A2)
{
    Z1 = W1 * X;
    for (int c = 0; c < Z1.cols(); c++)
    {
        Z1.col(c) = Z1.col(c) - b1;
    }
    A1 = ReLU(Z1);

    Z2 = W2 * A1;
    for (int c = 0; c < Z2.cols(); c++)
    {
        Z2.col(c) = Z2.col(c) - b2;
    }
    A2 = Softmax(Z2);
}

void init_params(int categories,
                 int num_features,
                 Eigen::MatrixXf &W1,
                 Eigen::VectorXf &b1,
                 Eigen::MatrixXf &W2,
                 Eigen::VectorXf &b2)
{
    // Random generates [-1:1]. Numpy is [0:1]
    W1 = Eigen::MatrixXf::Random(categories, num_features);
    W1 = W1.array() / 2.0f;
    b1 = Eigen::VectorXf::Random(categories);
    b1 = b1.array() / 2.0f;
    W2 = Eigen::MatrixXf::Random(categories, categories);
    W2 = W2.array() / 2.0f;
    b2 = Eigen::VectorXf::Random(categories, 1);
    b2 = b2.array() / 2.0f;
}

Eigen::MatrixXf one_hot_encode(Eigen::VectorXf &Z)
{
    Eigen::MatrixXf o = Eigen::MatrixXf::Zero(Z.rows(), Z.maxCoeff() + 1);

    for (int r = 0; r < Z.rows() - 1; r++)
    {
        o(r, int(Z(r))) = 1;
    }
    return o.transpose();
}

void back_prop(Eigen::MatrixXf &Z1,
               Eigen::MatrixXf &A1,
               Eigen::MatrixXf &Z2,
               Eigen::MatrixXf &A2,
               Eigen::MatrixXf &W2,
               Eigen::MatrixXf &X,
               Eigen::VectorXf &Y,
               Eigen::MatrixXf &dW1,
               float &db1,
               Eigen::MatrixXf &dW2,
               float &db2,
               Eigen::MatrixXf &one_hot_Y)
{
    int y_size = Y.rows();

    Eigen::MatrixXf dZ2 = A2 - one_hot_Y;
    dW2 = dZ2 * A1.transpose() / y_size;
    db2 = dZ2.sum() / y_size;

    Eigen::MatrixXf dZ1 = (W2.transpose() * dZ2).cwiseProduct(deriv_ReLU(Z1));
    dW1 = dZ1 * X.transpose() / y_size;
    db1 = dZ1.sum() / y_size;
}

Eigen::VectorXf get_predictions(Eigen::MatrixXf &P)
{
    Eigen::VectorXf p = Eigen::VectorXf::Zero(P.cols()).array() - 1;
    Eigen::Index maxIndex;
    for (int c = 0; c < P.cols(); c++)
    {
        P.col(c).maxCoeff(&maxIndex);
        p(c) = maxIndex;
    }
    return p;
}

int get_correct_prediction(Eigen::VectorXf &p, Eigen::VectorXf &y)
{
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> e = p.cwiseEqual(y);
    Eigen::VectorXi e_int = e.unaryExpr([](const bool x)
                                        { return x ? 1 : 0; });
    return e_int.sum();
}

float get_accuracy(int correct_prediction, int size)
{
    return 1.0f * correct_prediction / size;
}

void update_params(Eigen::MatrixXf &W1,
                   Eigen::VectorXf &b1,
                   Eigen::MatrixXf &W2,
                   Eigen::VectorXf &b2,
                   Eigen::MatrixXf &dW1,
                   float &db1,
                   Eigen::MatrixXf &dW2,
                   float &db2,
                   float &alpha)
{
    W1 = W1 - dW1 * alpha;
    b1 = b1.array() - db1 * alpha;
    W2 = W2 - dW2 * alpha;
    b2 = b2.array() - db2 * alpha;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "Wrong parameters: converter_mnist GENERATIONS MAX_ITEMS SAVE_IMG ALPHA" << std::endl;
        return EXIT_FAILURE;
    }

    std::srand((unsigned int)time(0));

    int num_generations = atoi(argv[1]);
    int max_items = atoi(argv[2]);
    bool save_img = (bool)atoi(argv[3]);
    float alpha = atof(argv[4]);

    std::string base_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST";
    std::string save_dir = "/media/brccabral/Data/CPP_Projects/CPP_Python_MNIST/MNIST/train";
    std::string img_path = base_dir + "/train-images.idx3-ubyte";
    std::string label_path = base_dir + "/train-labels.idx1-ubyte";

    std::vector<MNIST_Image> dataset;
    dataset = read_mnist_db(img_path.c_str(), label_path.c_str(), max_items, save_dir.c_str(), save_img);

    Eigen::MatrixXf mat = to_matrix(&dataset);

    Eigen::MatrixXf X_train = mat.leftCols(mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_train = mat.rightCols(1);             // n,1
    X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    Eigen::MatrixXf X = X_train.transpose();

    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf b1, b2;

    init_params(categories, X_train.cols(), W1, b1, W2, b2);
    Eigen::MatrixXf one_hot_Y = one_hot_encode(Y_train);

    Eigen::MatrixXf Z1, A1, Z2, A2;
    Eigen::MatrixXf dW1, dW2;
    float db1, db2;

    int correct_prediction = 0;
    float acc = 0.0f;

    for (int generation = 0; generation < num_generations; generation++)
    {
        forward_prop(W1, b1, W2, b2, X, Z1, A1, Z2, A2);

        back_prop(Z1, A1, Z2, A2, W2, X, Y_train, dW1, db1, dW2, db2, one_hot_Y);

        update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);

        if (generation % 50 == 0)
        {
            Eigen::VectorXf prediction = get_predictions(A2);
            correct_prediction = get_correct_prediction(prediction, Y_train);
            acc = get_accuracy(correct_prediction, Y_train.rows());
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }
    }
    Eigen::VectorXf prediction = get_predictions(A2);
    correct_prediction = get_correct_prediction(prediction, Y_train);
    acc = get_accuracy(correct_prediction, Y_train.rows());
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    for (auto &d : dataset)
    {
        delete[] d._pixels;
    }
}