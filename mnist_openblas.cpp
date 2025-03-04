#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <NeuralNet/NeuralNetOpenBLAS.h>

#define PRINT_VAR(x) #x << "=" << x

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

MatrixDouble *to_openblas(const std::vector<MNIST_Image> &_images)
{
    const int number_images = _images.size();
    const int number_pixels = _images.at(0)._rows * _images.at(0)._cols;

    auto *mat = create_matrix(number_images, number_pixels + 1);

    for (int img = 0; img < mat->rows; img++)
    {
        mat->data[img * mat->cols + 0] = float(_images.at(img)._label);
        for (int pix = 0; pix < number_pixels; pix++)
        {
            mat->data[img * mat->cols + pix + 1] = (unsigned char) _images.at(img)._pixels[pix];
        }
    }
    return mat;
}

MatrixDouble *get_Y(const MatrixDouble *mat)
{
    auto *Y = create_matrix(mat->rows, 1);
    cblas_dcopy(mat->rows, mat->data, mat->cols, Y->data, 1);
    return Y;
}

MatrixDouble *get_X(const MatrixDouble *mat)
{
    auto *X = create_matrix(mat->rows, mat->cols - 1);

    for (int row = 0; row < mat->rows; row++)
    {
        cblas_dcopy(X->cols, &mat->data[row * mat->cols + 1], 1, &X->data[row], 1);
    }

    return X;
}

int main()
{
    omp_set_num_threads(std::max(omp_get_max_threads() - 2, 1));

    CSimpleIniA ini;
    ini.SetUnicode();

    const SI_Error rc = ini.LoadFile("config.ini");
    if (rc < 0)
    {
        std::cout << "Error loading config.ini" << std::endl;
        return EXIT_FAILURE;
    }
    SI_ASSERT(rc == SI_OK);

    const auto num_generations = (int) ini.GetLongValue("MNIST", "GENERATIONS", 5);
    const auto max_items = (int) ini.GetLongValue("MNIST", "MAX_ITEMS", 15);
    const bool save_img = ini.GetBoolValue("MNIST", "SAVE_IMG", false);
    const auto alpha = (float) ini.GetDoubleValue("MNIST", "ALPHA", 0.1);
    const auto hidden_layer_size = (int) ini.GetLongValue("MNIST", "HIDDEN_LAYER_SIZE", 10);

    const std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    const std::string save_dir = base_dir + "/train";
    const std::string img_filename =
            ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    const std::string img_path = base_dir + "/" + img_filename;
    const std::string label_filename =
            ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    const std::string label_path = base_dir + "/" + label_filename;

    std::cout << PRINT_VAR(num_generations) << " " << PRINT_VAR(max_items) << " "
              << PRINT_VAR(save_img) << " " << PRINT_VAR(alpha) << " "
              << PRINT_VAR(hidden_layer_size) << " " << PRINT_VAR(base_dir) << " "
              << PRINT_VAR(save_dir) << " " << PRINT_VAR(img_filename) << " " << PRINT_VAR(img_path)
              << " " << PRINT_VAR(label_filename) << " " << PRINT_VAR(label_path) << " "
              << std::endl;

    MNIST_Dataset train_dataset(
            img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);
    std::cout << PRINT_VAR(train_dataset.get_images_length()) << std::endl;
    std::cout << PRINT_VAR(train_dataset.get_label_from_index(4)) << std::endl;

#ifdef CV_SAVE_IMAGES
    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);
#endif // CV_SAVE_IMAGES

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    auto train_mat = to_openblas(train_dataset._images);

    auto Y_train_float = get_Y(train_mat);
    auto X_train = get_X(train_mat);
    cblas_dscal(X_train->rows * X_train->cols, 1 / 255.0, X_train->data, 1);

    printf("%g\n", Y_train_float->data[4 * Y_train_float->cols + 0]);
    DESCRIBE_MATRIX(train_mat);
    DESCRIBE_MATRIX(Y_train_float);
    DESCRIBE_MATRIX(X_train);
    for (int c = 0; c < X_train->cols; ++c)
    {
        printf("%g ", X_train->data[4 * X_train->cols + c]);
    }
    printf("\n");

    double categories =
            Y_train_float->data[cblas_idmax(Y_train_float->rows, Y_train_float->data, 1)] + 1;
    printf("%g\n", categories);

    // cblas does not have a Transpose function, but when multiplying matrices, we tell the mul()
    // that X is transposed

    auto *neural_net = create_neuralnet_openblas(X_train->cols, hidden_layer_size, categories);
    auto *one_hot_Y = one_hot_encode(Y_train_float, 0);

    MatrixDouble *output;

    int correct_prediction = 0;
    float acc = 0.0f;

    auto *prediction = create_matrix(X_train->rows, 1);

    DESCRIBE_NN(neural_net);

    for (int generation = 0; generation < num_generations; generation++)
    {
        output = forward_prop(neural_net, X_train);

        if (generation % 50 == 0)
        {
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }
    }

    free_matrix(X_train);
    free_matrix(Y_train_float);
    free_matrix(train_mat);
    free_matrix(one_hot_Y);
    free_matrix(prediction);
    free_neuralnet_openblas(neural_net);
    return 0;
}
