#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <SimpleIni/SimpleIni.h>

#ifdef DEBUG
#define DUMP_VAR(x) std::cout << "---\n" \
                              << #x "\n" \
                              << x << "\n---" << std::endl
#else
#define DUMP_VAR(x)
#endif

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

float get_float_matrix(Eigen::MatrixXf &M, int r, int c)
{
    return M(r, c);
}

float get_float_vector(Eigen::VectorXf &V, int r)
{
    return V(r);
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
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
    MNIST_Image(uint32_t rows, uint32_t cols, int label, char *pixels, int item_id)
        : _rows(rows), _cols(cols), _label(label), _db_item_id(item_id)
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
    void save_as_csv(std::string save_filename)
    {
        std::ofstream outfile;
        if (_db_item_id == 0)
            outfile.open(save_filename);
        else
            outfile.open(save_filename, std::ios_base::app);

        outfile << _label;
        for (int p = 0; p < _rows * _cols; p++)
        {
            outfile << ',' << std::to_string((unsigned char)_pixels[p]);
        }
        outfile << "\n";
        outfile.close();
    }
};

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
                  int label_magic)
        : _image_filename(image_filename), _label_filename(label_filename), _image_magic(image_magic), _label_magic(label_magic)
    {
    }
    ~MNIST_Dataset()
    {
        for (auto &img : _images)
        {
            delete[] img._pixels;
        }
    }

    void save_dataset_as_png(std::string save_dir)
    {
        for (MNIST_Image img : _images)
        {
            img.save_as_png(save_dir);
        }
    };

    void save_dataset_as_csv(std::string save_dir)
    {
        for (MNIST_Image img : _images)
        {
            img.save_as_csv(save_dir);
        }
    };

    Eigen::MatrixXf to_matrix()
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
    }

    void read_mnist_db(const int max_items)
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

class NeuralNet
{
private:
    // layers
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::MatrixXf W2;
    Eigen::VectorXf b2;

    // back prop
    Eigen::MatrixXf Z1;
    Eigen::MatrixXf A1;
    Eigen::MatrixXf Z2;
    Eigen::MatrixXf A2;

    // gradients
    Eigen::MatrixXf dW1;
    float db1;
    Eigen::MatrixXf dW2;
    float db2;

public:
    NeuralNet(int hidden_layer_size,
              int categories,
              int num_features)
    {
        // Random generates [-1:1]. Numpy is [0:1]
        W1 = Eigen::MatrixXf::Random(hidden_layer_size, num_features);
        W1 = W1.array() / 2.0f;
        b1 = Eigen::VectorXf::Random(hidden_layer_size);
        b1 = b1.array() / 2.0f;
        W2 = Eigen::MatrixXf::Random(categories, hidden_layer_size);
        W2 = W2.array() / 2.0f;
        b2 = Eigen::VectorXf::Random(categories, 1);
        b2 = b2.array() / 2.0f;
    }

    static Eigen::MatrixXf ReLU(Eigen::MatrixXf &Z)
    {
        return Z.cwiseMax(0);
    }

    static Eigen::MatrixXf Softmax(Eigen::MatrixXf &Z)
    {
        Eigen::MatrixXf e = Z.array().exp();
        Eigen::MatrixXf s = e.colwise().sum();
        for (int c = 0; c < e.cols(); c++)
        {
            e.col(c) = e.col(c) / s(c);
        }
        return e;
    }

    Eigen::MatrixXf forward_prop(Eigen::MatrixXf &X)
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

        return A2;
    }

    static Eigen::MatrixXf one_hot_encode(Eigen::VectorXf &Z)
    {
        Eigen::MatrixXf o = Eigen::MatrixXf::Zero(Z.rows(), Z.maxCoeff() + 1);

        for (int r = 0; r < Z.rows() - 1; r++)
        {
            o(r, int(Z(r))) = 1;
        }
        return o.transpose();
    }

    static Eigen::MatrixXf deriv_ReLU(Eigen::MatrixXf &Z)
    {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> b2 = (Z.array() > 0);
        return b2.unaryExpr([](const bool x)
                            { return x ? 1.0f : 0.0f; });
    }

    void back_prop(
        Eigen::MatrixXf &X,
        Eigen::VectorXf &Y,
        Eigen::MatrixXf &one_hot_Y,
        float alpha)
    {
        int y_size = Y.rows();

        Eigen::MatrixXf dZ2 = A2 - one_hot_Y;
        dW2 = dZ2 * A1.transpose() / y_size;
        db2 = dZ2.sum() / y_size;

        Eigen::MatrixXf dZ1 = (W2.transpose() * dZ2).cwiseProduct(deriv_ReLU(Z1));
        dW1 = dZ1 * X.transpose() / y_size;
        db1 = dZ1.sum() / y_size;

        W1 = W1 - dW1 * alpha;
        b1 = b1.array() - db1 * alpha;
        W2 = W2 - dW2 * alpha;
        b2 = b2.array() - db2 * alpha;
    }

    static Eigen::VectorXf get_predictions(Eigen::MatrixXf &P)
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

    static int get_correct_prediction(Eigen::VectorXf &p, Eigen::VectorXf &y)
    {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> e = p.cwiseEqual(y);
        Eigen::VectorXi e_int = e.unaryExpr([](const bool x)
                                            { return x ? 1 : 0; });
        return e_int.sum();
    }

    static float get_accuracy(int correct_prediction, int size)
    {
        return 1.0f * correct_prediction / size;
    }
};

int main(int argc, char *argv[])
{
    std::srand((unsigned int)time(0));

    CSimpleIniA ini;
    ini.SetUnicode();

    SI_Error rc = ini.LoadFile("../../config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    };
    SI_ASSERT(rc == SI_OK);

    int num_generations = ini.GetLongValue("MNIST", "GENERATIONS", 5);
    int max_items = ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    float alpha = ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    int hidden_layer_size = ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST");
    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images.idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename = ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels.idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

    MNIST_Dataset train_dataset(img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    Eigen::MatrixXf train_mat = train_dataset.to_matrix();

    Eigen::MatrixXf X_train = train_mat.leftCols(train_mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_train = train_mat.rightCols(1);                   // n,1
    X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    Eigen::MatrixXf X_train_T = X_train.transpose();

    NeuralNet neural_net(hidden_layer_size, categories, X_train.cols());
    Eigen::MatrixXf one_hot_Y = NeuralNet::one_hot_encode(Y_train);

    Eigen::MatrixXf output;

    int correct_prediction = 0;
    float acc = 0.0f;

    for (int generation = 0; generation < num_generations; generation++)
    {
        output = neural_net.forward_prop(X_train_T);

        if (generation % 50 == 0)
        {
            Eigen::VectorXf prediction = NeuralNet::get_predictions(output);
            correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_train);
            acc = NeuralNet::get_accuracy(correct_prediction, Y_train.rows());
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }

        neural_net.back_prop(X_train_T, Y_train, one_hot_Y, alpha);
    }
    Eigen::VectorXf prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_train);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_train.rows());
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    save_dir = base_dir + "/test";
    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images.idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels.idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    MNIST_Dataset test_dataset(img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);

    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");

    Eigen::MatrixXf test_mat = test_dataset.to_matrix();

    Eigen::MatrixXf X_test = test_mat.leftCols(test_mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_test = test_mat.rightCols(1);                  // n,1
    X_test = X_test / 255.0;

    Eigen::MatrixXf X_test_T = X_test.transpose();

    output = neural_net.forward_prop(X_test_T);

    prediction = NeuralNet::get_predictions(output);
    correct_prediction = NeuralNet::get_correct_prediction(prediction, Y_test);
    acc = NeuralNet::get_accuracy(correct_prediction, Y_test.rows());
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);
}