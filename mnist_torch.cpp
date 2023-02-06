#include <SimpleIni/SimpleIni.h>
#include <MNIST/MNIST_Dataset.hpp>
#include <torch/torch.h>
#include <TorchNet/torchnet.hpp>

#define TRAIN_IMAGE_MAGIC 2051
#define TRAIN_LABEL_MAGIC 2049
#define TEST_IMAGE_MAGIC 2051
#define TEST_LABEL_MAGIC 2049

torch::Tensor eigenMatrixToTorchTensor(Eigen::MatrixXf e){
    auto t = torch::empty({e.cols(),e.rows()});
    float *data = t.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> ef(data, t.size(1), t.size(0));
    ef = e.cast<float>();
    t.requires_grad_(true);
    return t.transpose(0,1);
}

torch::Tensor eigenVectorToTorchTensor(Eigen::VectorXf e){
    auto t = torch::rand({e.rows()});
    float *data = t.data_ptr<float>();

    Eigen::Map<Eigen::VectorXf> ef(data, t.size(0), 1);
    ef = e.cast<float>();

    t.requires_grad_(true);
    return t;
}



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

    std::cout << "Reading dataset file" << std::endl;
    MNIST_Dataset train_dataset(img_path.c_str(), label_path.c_str(), TRAIN_IMAGE_MAGIC, TRAIN_LABEL_MAGIC);
    train_dataset.read_mnist_db(max_items);

    if (save_img)
        train_dataset.save_dataset_as_png(save_dir);

    train_dataset.save_dataset_as_csv(save_dir + "/train.csv");

    std::cout << "Converting to matrix" << std::endl;
    Eigen::MatrixXf train_mat = train_dataset.to_matrix();

    Eigen::MatrixXf X_train = train_mat.leftCols(train_mat.cols() - 1); // n,784 = 28*28
    Eigen::VectorXf Y_train = train_mat.rightCols(1);                   // n,1
    // Eigen::VectorXi Y_train_int = Y_train.unaryExpr([](const float x)
    //                                                 { return int(x); });
    // X_train = X_train / 255.0;

    int categories = Y_train.maxCoeff() + 1;

    std::cout << "Preparing tensors" << std::endl;

    // torch::Tensor x_tensor = torch::from_blob(X_train.array().data(),
    //                                           {X_train.rows(), X_train.cols()}
    //                                         //   ,torch::requires_grad()
    //                                           )
    //                              .clone();
    // torch::Tensor y_tensor = torch::from_blob(Y_train.array().data(),
    //                                           Y_train.rows(), at::kLong)
    //                              .clone();
    
    // torch::Tensor x_tensor = torch::empty({X_train.rows(), X_train.cols()});
    // torch::Tensor y_tensor = torch::empty(Y_train.rows());

    // for(int r = 0 ; r < X_train.rows(); r++){
    //     for(int c = 0 ; c < X_train.cols(); c++){
    //         x_tensor.index_put_({r, c}, X_train(r, c));
    //     }
    //     y_tensor.index_put_({r}, Y_train(r));

    //     if (r % 1000 == 0)
    //         std::cout << "Row " << r << std::endl;
    // }
    // torch::Tensor y_tensor_i = y_tensor.toType(c10::ScalarType::Long);

    torch::Tensor x_tensor = eigenMatrixToTorchTensor(X_train);
    torch::Tensor y_tensor = eigenVectorToTorchTensor(Y_train);

    std::cout << x_tensor.sizes() << std::endl;
    std::cout << x_tensor.index({0}) << std::endl;
    std::cout << y_tensor.sizes() << std::endl;
    std::cout << y_tensor.index({0}) << std::endl;

    Net net(X_train.cols(), hidden_layer_size, categories);
    net->train();

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/alpha);
    torch::Tensor values, indices, prediction, correct_bool;
    std::tuple<torch::Tensor, torch::Tensor> tm;
    int correct_prediction = 0;
    float acc = 0.0f;

    for (int generation = 0; generation < num_generations; generation++)
    {
        optimizer.zero_grad();
        torch::Tensor prediction = net->forward(x_tensor);
        torch::Tensor loss = torch::nll_loss(prediction, y_tensor);
        loss.backward();
        optimizer.step();
        
        if (generation % 50 == 0)
        {
            tm = torch::max(prediction, 1);
            std::tie(values, indices) = tm;

            correct_bool = y_tensor == indices;
            correct_prediction = correct_bool.sum().item<int>();
            
            acc = 1.0f * correct_prediction / Y_train.rows();
            printf("Generation %d\t Correct %d\tAccuracy %.4f\n", generation, correct_prediction, acc);
        }
    }
    prediction = net->forward(x_tensor);
    
    tm = torch::max(prediction, 1);
    std::tie(values, indices) = tm;

    correct_bool = y_tensor == indices;
    correct_prediction = correct_bool.sum().item<int>();
    
    acc = 1.0f * correct_prediction / Y_train.rows();
    printf("Final \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);


    // net->eval();
    // MNIST_Dataset test_dataset(img_path.c_str(), label_path.c_str(), TEST_IMAGE_MAGIC, TEST_LABEL_MAGIC);
    // test_dataset.read_mnist_db(max_items);

    // if (save_img)
    //     test_dataset.save_dataset_as_png(save_dir);

    // test_dataset.save_dataset_as_csv(save_dir + "/test.csv");

    // Eigen::MatrixXf test_mat = test_dataset.to_matrix();

    // Eigen::MatrixXf X_test = test_mat.leftCols(test_mat.cols() - 1); // n,784 = 28*28
    // Eigen::VectorXf Y_test = test_mat.rightCols(1);                  // n,1
    // // Eigen::VectorXi Y_test_int = Y_test.unaryExpr([](const float x)
    // //                                                 { return int(x); });
    // X_test = X_test / 255.0;

    // x_tensor = torch::from_blob(X_test.data(),
    //                             {X_test.rows(), X_test.cols()},
    //                             torch::requires_grad())
    //                .clone();
    // y_tensor = torch::from_blob(Y_test.data(),
    //                             Y_test.rows())
    //                .clone();
    // y_tensor_i = y_tensor.toType(c10::ScalarType::Long);

    // prediction = net->forward(x_tensor);
    
    // tm = torch::max(prediction, 1);
    // std::tie(values, indices) = tm;

    // correct_bool = y_tensor_i == indices;
    // correct_prediction = correct_bool.sum().item<int>();
    
    // acc = 1.0f * correct_prediction / Y_train.rows();
    // printf("Test \t Correct %d\tAccuracy %.4f\n", correct_prediction, acc);



    return EXIT_SUCCESS;
}
