# MNIST database Neural Network in C++ and Python

This project aims to demonstrate a simple Neural Network using only math libraries (`numpy` for Python and `Eigen` for C++). It is not intent to use in production code.  

Download MNIST database from http://yann.lecun.com/exdb/mnist/ (**`HTTP`** only, in Firefox need to tweak `https` settings)  
Files: train images, train labels, test images, test labels  
Files are in `idx` format  

C++ code needs `Eigen` library as math library (matrices algebra) https://eigen.tuxfamily.org/. I used `Eigen 3.4.0`.  

C++ code needs `OpenCV` libraries to save images on disk https://docs.opencv.org/4.6.0/d7/d9f/tutorial_linux_install.html. I used `OpenCV 4.6.0`. There is no need to compile all opencv shared libraries, only `opencv_core`, `opencv_imgcodecs` and `opencv_imgproc` (dependency of _imgcodecs_).  

C++ code needs `SimpleIni` include files to parse and use `config.ini` https://github.com/brofield/simpleini/. I used `SimpleInit 4.19`.

C++ code to read MNIST database was adapted from https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c  

Python code uses `numpy` as math library (matrices algebra) https://numpy.org/. I used `numpy 1.24.1`.  

Python code uses `pillow` to save images to disk https://pillow.readthedocs.io/en/stable/. I used `pillow 9.4.0`.  

```
pip install pillow==9.4.0
pip install numpy==1.24.1
```

Python code to read MNIST database was adapted from https://jamesmccaffrey.wordpress.com/2020/05/05/converting-raw-mnist-binary-files-to-text-files/  

Python code for the Neural Network was adapted from a tutorial from Samson Zhang [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://www.youtube.com/watch?v=w8yWXqWQYmU)  

C++ code for the Neural Network is an adaptation from myself based on the Python code.  

C++ Libtorch
- Download `libtorch` : PyTorch C++ library C++ ABI  
    - I downloaded CPU version 1.13.0, GPU is 1.8 GB compacted. Also, to use GPU need to install CUDA libs 7 GB.  
    - https://pytorch.org/get-started/locally/  
    - libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip  
    - do NOT download the "pre-cxx11 ABI"  
    - the CXX_FLAGS needs to have "-D_GLIBCXX_USE_CXX11_ABI=1", not 0  
    - Unzip to this workspace (`./workspace/libtorch`), or somewhere in your path (`/usr/local` or `$HOME/.local`)

Python PyTorch

Install pytorch from offical website https://pytorch.org/get-started/locally/  
For this project I used `torch==1.13.1`, `torchvision==0.14.1`, `torchaudio==0.13.1`, but this project doesn't have any example of `torchvision` or `torchaudio`.

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```