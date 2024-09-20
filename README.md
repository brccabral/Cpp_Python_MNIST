# MNIST database Neural Network in C++ and Python

This project aims to demonstrate a simple Neural Network using only math libraries (`numpy` for Python and `Eigen`, `NumCpp` and `xtensor` for C++). It is not intent to use in production code.  
I created a YouTube playlist demonstrating the project https://www.youtube.com/watch?v=61qSfHAv9I0&list=PLZGFbvBWrR0HX92alWRmZQTrIOdDKVsKP&index=1  

## Dataset
EMNIST dataset from https://www.nist.gov/itl/products-and-services/emnist-dataset has 240000 images, but C++ PyTorch has hardcoded to load only the 60000 dataset. The solution is to use the Python's torch library to download the 60000 dataset (run `download.py`). The 240000 dataset can be used in Python script, but not in C++.

Files: train images, train labels, test images, test labels   

## Libraries
- C++
  - **Eigen**  
    https://eigen.tuxfamily.org/  
    Math library (matrices algebra)  
    I used `Eigen 3.4.0`.  
  - **OpenCV**  
    https://docs.opencv.org/4.6.0/d7/d9f/tutorial_linux_install.html  
    Library to save images on disk (not used to train/test)  
    I used `OpenCV 4.6.0`. There is no need to compile all opencv shared libraries, only `opencv_core`, `opencv_imgcodecs` and `opencv_imgproc` (dependency of _imgcodecs_).  
  - **SimpleIni**  
    https://github.com/brofield/simpleini/  
    Parse and use `config.ini`  
    Header only library  
    I used `SimpleInit 4.19`.  
  - **NumCpp**  
    https://github.com/dpilger26/NumCpp  
    Another math library (similar to `numpy`)  
    Header only library  
  - **xtensor**  
    https://github.com/xtensor-stack/xtensor  
    One more math library (similar to `numpy`)  
    Header only library  
  - **C++ Libtorch**  
    https://pytorch.org/get-started/locally/  
    Download `libtorch` : PyTorch C++ library C++ ABI  
    I downloaded CPU version 2.0.1, GPU is 1.8 GB compacted. Also, to use GPU need to install CUDA libs 7 GB.  
    libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip  
    do NOT download the "pre-cxx11 ABI"  
    the CXX_FLAGS needs to have "-D_GLIBCXX_USE_CXX11_ABI=1", not 0  
    Unzip to this workspace (`./workspace/libtorch`), or somewhere in your path (`/usr/local` or `$HOME/.local`)

- Python
  - **Python PyTorch**  
    https://pytorch.org/get-started/locally/  
    For this project I used `torch==2.0.1`, `torchvision==0.15.2`, `torchaudio==2.0.2`, but this project doesn't have any example of `torchaudio`, and `torchvision` is just to load the MNIST dataset.  
    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    Or
    ```shell
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
    ```
    The `--index-url` is used to download the CPU version. If not passed as argument, it will download the GPU version (+4GB).
    It will download and install `numpy` and `pillow`. In my case it was downloaded `numpy 1.24.1` and `pillow 9.3.0`.  
  - **numpy**  
    https://numpy.org/  
    Math library (matrices algebra)
  - **pillow**  
    https://pillow.readthedocs.io/en/stable/  
    Library to save images to disk (not used to train/test)  

## References

C++ code that reads MNIST database was adapted from https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c  

Python code that reads MNIST database was adapted from https://jamesmccaffrey.wordpress.com/2020/05/05/converting-raw-mnist-binary-files-to-text-files/  

Python code for the Neural Network was adapted from a tutorial from Samson Zhang [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://www.youtube.com/watch?v=w8yWXqWQYmU)  

C++ code for the Neural Network is an adaptation from myself based on the Zhang's Python code.  

Migrate from `numpy` to `NumCpp` https://dpilger26.github.io/NumCpp/doxygen/html/index.html  

Migrate from `numpy` to `xtensor` https://xtensor.readthedocs.io/en/latest/numpy.html  

PyTorch/Libtorch examples were adapted from official doc.  
Python https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html  
C++ https://pytorch.org/cppdocs/frontend.html#end-to-end-example  

Libtorch tutorial from Alan Tessier https://www.youtube.com/watch?v=RFq8HweBjHA&list=PLZAGo22la5t4UWx37MQDpXPFX3rTOGO3k&index=1  

Helper function to convert Eigen to Torch
https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156/6

## Windows
On Windows I could only build the project in Visual Studio cmake generator, couldn't make it work with MinGW/Unix Makefiles. OpenCV also needs to be built with Visual Studio cmake generator.
