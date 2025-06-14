#if _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <ctime>
#include <iostream>
#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <NeuralNetOpenBLAS/NeuralNetOpenBLAS.h>

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
    if (mat == NULL)
    {
        return NULL;
    }

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
    if (mat == NULL)
    {
        return NULL;
    }
    auto *Y = create_matrix(mat->rows, 1);
    if (Y == NULL)
    {
        return NULL;
    }
    cblas_dcopy(mat->rows, mat->data, mat->cols, Y->data, 1);
    return Y;
}

MatrixDouble *get_X(const MatrixDouble *mat)
{
    if (mat == NULL)
    {
        return NULL;
    }
    auto *X = create_matrix(mat->rows, mat->cols - 1);
    if (X == NULL)
    {
        return NULL;
    }

    for (int row = 0; row < mat->rows; row++)
    {
        cblas_dcopy(X->cols, &mat->data[row * mat->cols + 1], 1, &X->data[row * X->cols], 1);
    }

    return X;
}

int main()
{
    nn_seed(time(NULL));

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

    std::string base_dir = ini.GetValue("MNIST", "BASE_DIR", "MNIST_data/MNIST/raw");
    std::string save_dir = base_dir + "/train";
    std::string img_filename = ini.GetValue("MNIST", "TRAIN_IMAGE_FILE", "train-images-idx3-ubyte");
    std::string img_path = base_dir + "/" + img_filename;
    std::string label_filename =
            ini.GetValue("MNIST", "TRAIN_LABEL_FILE", "train-labels-idx1-ubyte");
    std::string label_path = base_dir + "/" + label_filename;

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
    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");
#endif // CV_SAVE_IMAGES

    auto train_mat = to_openblas(train_dataset._images);
    if (train_mat == NULL)
    {
        printf("Failed to convert dataset to OpenBLAS\n");
        exit(EXIT_FAILURE);
    }

    auto *Y_train_float = get_Y(train_mat);
    if (Y_train_float == NULL)
    {
        printf("Failed to convert get Y from dataset\n");
        free_matrix(train_mat);
        exit(EXIT_FAILURE);
    }
    auto *X_train = get_X(train_mat);
    if (X_train == NULL)
    {
        printf("Failed to convert get X from dataset\n");
        free_matrix(train_mat);
        free_matrix(Y_train_float);
        exit(EXIT_FAILURE);
    }

    printf("%g\n", Y_train_float->data[4 * Y_train_float->cols + 0]);
    DESCRIBE_MATRIX(train_mat);
    DESCRIBE_MATRIX(Y_train_float);
    DESCRIBE_MATRIX(X_train);
    for (int c = 0; c < X_train->cols; ++c)
    {
        printf("%g ", X_train->data[4 * X_train->cols + c]);
    }
    printf("\n");

    const double max_X = X_train->data[cblas_idmax(X_train->rows, X_train->data, 1)];
    cblas_dscal(X_train->rows * X_train->cols, 1.0 / max_X, X_train->data, 1);

    const double categories =
            Y_train_float->data[cblas_idmax(Y_train_float->rows, Y_train_float->data, 1)] + 1;
    printf("%g\n", categories);

    // cblas does not have a Transpose function, but when multiplying matrices, we tell the
    // mul() that X is transposed

    auto *neural_net = create_neuralnet_openblas(X_train->cols, hidden_layer_size, categories);
    if (neural_net == NULL)
    {
        printf("Failed create neural net\n");
        free_matrix(train_mat);
        free_matrix(Y_train_float);
        free_matrix(X_train);
        exit(EXIT_FAILURE);
    }
    auto *one_hot_Y = one_hot_encode(Y_train_float, 0);
    if (one_hot_Y == NULL)
    {
        printf("Failed to get One Hot Encode\n");
        free_matrix(train_mat);
        free_matrix(Y_train_float);
        free_matrix(X_train);
        free_neuralnet_openblas(neural_net);
        exit(EXIT_FAILURE);
    }

    int correct_prediction = 0;
    float acc = 0.0f;

    DESCRIBE_NN(neural_net);

    for (int generation = 0; generation < num_generations; generation++)
    {
        forward_prop(neural_net, X_train);

        if (generation % 50 == 0)
        {
            get_predictions(neural_net);
            correct_prediction = get_correct_prediction(neural_net, Y_train_float);
            acc = 1.0 * correct_prediction / Y_train_float->rows;
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction,
                   acc);
        }

        back_prop(neural_net, X_train, one_hot_Y, alpha);
    }
    forward_prop(neural_net, X_train);
    get_predictions(neural_net);
    correct_prediction = get_correct_prediction(neural_net, Y_train_float);
    acc = 1.0 * correct_prediction / Y_train_float->rows;
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    img_filename = ini.GetValue("MNIST", "TEST_IMAGE_FILE", "t10k-images-idx3-ubyte");
    img_path = base_dir + "/" + img_filename;
    label_filename = ini.GetValue("MNIST", "TEST_LABEL_FILE", "t10k-labels-idx1-ubyte");
    label_path = base_dir + "/" + label_filename;

    MNIST_Dataset test_dataset(
            img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    test_dataset.read_mnist_db(max_items);

#ifdef CV_SAVE_IMAGES
    save_dir = base_dir + "/test";
    if (save_img)
        test_dataset.save_dataset_as_png(save_dir);
    test_dataset.save_dataset_as_csv(save_dir + "/test.csv");
#endif // CV_SAVE_IMAGES


    auto test_mat = to_openblas(test_dataset._images);
    if (test_mat == NULL)
    {
        printf("Failed to convert dataset to OpenBLAS\n");
        exit(EXIT_FAILURE);
    }

    auto *Y_test_float = get_Y(test_mat);
    if (Y_test_float == NULL)
    {
        printf("Failed to convert get Y from dataset\n");
        free_matrix(test_mat);
        exit(EXIT_FAILURE);
    }
    auto *X_test = get_X(test_mat);
    if (X_test == NULL)
    {
        printf("Failed to convert get X from dataset\n");
        free_matrix(test_mat);
        free_matrix(Y_test_float);
        exit(EXIT_FAILURE);
    }

    cblas_dscal(X_test->rows * X_test->cols, 1.0 / max_X, X_test->data, 1);

    forward_prop(neural_net, X_test);
    get_predictions(neural_net);
    correct_prediction = get_correct_prediction(neural_net, Y_test_float);
    acc = 1.0 * correct_prediction / Y_test_float->rows;
    printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);

    free_matrix(X_train);
    free_matrix(Y_train_float);
    free_matrix(train_mat);
    free_matrix(X_test);
    free_matrix(Y_test_float);
    free_matrix(test_mat);
    free_matrix(one_hot_Y);
    free_neuralnet_openblas(neural_net);
    return 0;
}
