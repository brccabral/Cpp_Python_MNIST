# MNIST database Neural Network in C++ and Python

This project aims to demonstrate a simple Neural Network using only math libraries (`numpy` for Python and `Eigen`,
`NumCpp` and `xtensor` for C++). It is not intent to use in production code.  
I created a YouTube playlist demonstrating the
project https://www.youtube.com/watch?v=61qSfHAv9I0&list=PLZGFbvBWrR0HX92alWRmZQTrIOdDKVsKP&index=1

## Dataset

EMNIST dataset from https://www.nist.gov/itl/products-and-services/emnist-dataset has 240000 images, but C++ PyTorch has
hardcoded to load only the 60000 dataset. The solution is to use the Python's torch library to download the 60000
dataset (run `download.py`). The 240000 dataset can be used in Python script, but not in C++.

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
      I used `OpenCV 4.6.0`. There is no need to compile all opencv shared libraries, only `opencv_core`,
      `opencv_imgcodecs` and `opencv_imgproc` (dependency of _imgcodecs_).  
      On Windows, you may need to compile both `Debug` and `Release`.
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
      ```shell
      sudo apt install xtensor-dev
      ```
      To enable SIMD optimizations, need to install **xsimd**  
      To enable parallel assignment loop, need to install **tbb** (Linux) or **oneTBB** (Windows).  
      On Windows, run `cmake *** --target install` and add to `%PATH%`  
      On Windows, check if this PR has been merged https://github.com/xtensor-stack/xtensor/pull/2799
    - **xtensor-blas**  
      https://github.com/xtensor-stack/xtensor-blas  
      An _xtensor_ extension for linear algebra _BLAS (Basic Linear Algebra Subprograms)_  
      Header only library
      ```shell
      sudo apt install libxtensor-blas-dev
      ```
      On Windows, we need `OpenBLAS`  
      On Windows, run `cmake *** --target install` and add to `%PATH%`
    - **xsimd**  
      https://github.com/xtensor-stack/xsimd  
      SIMD optimizations for xtensor
      ```shell
      sudo apt install libxsimd-dev
      ```
      ```shell
      git clone --depth 1 https://github.com/xtensor-stack/xsimd.git
      cd xsimd
      mkdir cmake-build-release
      cd cmake-build-release
      cmake -DCMAKE_BUILD_TYPE=Release ..
      sudo make install -j 10
      ```
      On Windows, run `cmake *** --target install` and add to `%PATH%`
    - **tbb**
      Allow parallelism/multi-threads for C++  
      On Windows, use **oneTBB**
      ```shell
      sudo apt install libtbb-dev
      ```
    - **oneTBB**  
      https://github.com/oneapi-src/oneTBB/  
      _oneTBB is a flexible C++ library that simplifies the work of adding parallelism to complex applications, even if
      you are not a threading expert._
      ```shell
      git clone --depth 1 https://github.com/oneapi-src/oneTBB.git
      cd oneTBB
      mkdir cmake-build-release
      cd cmake-build-release
      cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -DTBB_TEST:BOOL=OFF ..
      cmake --install .
      ```
      On Windows, run `cmake *** --target install` and add to `%PATH%` both installation folder and `*\bin`
    - **xtl**
      https://github.com/xtensor-stack/xtl  
      "stl" for "xtensor" (containers, algorithms)  
      I had to install on Windows, but on Linux it may be a dependency of previous packages.  
      On Windows, run `cmake *** --target install` and add to `%PATH%`
    - **C++ Libtorch**  
      https://pytorch.org/get-started/locally/  
      Download `libtorch` : PyTorch C++ library C++ ABI  
      `libtorch-cxx11-abi-shared-with-deps-1.13.0+cpu.zip`  
      I downloaded CPU version 2.0.1, GPU is 1.8 GB compacted. Also, to use GPU need to install CUDA libs 7 GB.  
      Do NOT download the "pre-cxx11 ABI".  
      The CXX_FLAGS needs to have "-D_GLIBCXX_USE_CXX11_ABI=1", not 0  
      Unzip to this workspace (`./workspace/libtorch`), or somewhere in your path (`/usr/local` or `$HOME/.local`).  
      On Windows, you may need to download both `Debug` and `Release`.
    - **OpenBLAS**
      https://www.openblas.net/  
      An implementation of _BLAS (Basic Linear Algebra Subprograms)_  
      I had to download it on Windows.  
      Add to your `%PATH%` and set `-DOpenBLAS_DIR` on _cmake_.
      On Linux it is a dependency of `xtensor`, but can be installed independently.
      ```shell
      sudo apt install libopenblas-dev liblapacke-dev
      ```

- Python
    - **Python PyTorch**  
      https://pytorch.org/get-started/locally/  
      For this project I used `torch==2.0.1`, `torchvision==0.15.2`, `torchaudio==2.0.2`, but this project doesn't have
      any example of `torchaudio`, and `torchvision` is just to load the MNIST dataset.
      ```shell
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ```
      Or
      ```shell
      pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
      ```
      The `--index-url` is used to download the CPU version. If not passed as argument, it will download the GPU
      version (+4GB).  
      It will download and install `numpy` and `pillow`. In my case it was downloaded `numpy 1.24.1` and `pillow 9.3.0`.
    - **numpy**  
      https://numpy.org/  
      Math library (matrices algebra)
    - **pillow**  
      https://pillow.readthedocs.io/en/stable/  
      Library to save images to disk (not used to train/test)

## References

C++ code that reads MNIST database was adapted
from https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c

Python code that reads MNIST database was adapted
from https://jamesmccaffrey.wordpress.com/2020/05/05/converting-raw-mnist-binary-files-to-text-files/

Python code for the Neural Network was adapted from a tutorial from Samson
Zhang [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://www.youtube.com/watch?v=w8yWXqWQYmU)

C++ code for the Neural Network is an adaptation from myself based on the Zhang's Python code.

Migrate from `numpy` to `NumCpp` https://dpilger26.github.io/NumCpp/doxygen/html/index.html

Migrate from `numpy` to `xtensor` https://xtensor.readthedocs.io/en/latest/numpy.html

PyTorch/Libtorch examples were adapted from official doc.  
Python https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html  
C++ https://pytorch.org/cppdocs/frontend.html#end-to-end-example

Libtorch tutorial from Alan
Tessier https://www.youtube.com/watch?v=RFq8HweBjHA&list=PLZAGo22la5t4UWx37MQDpXPFX3rTOGO3k&index=1

Helper function to convert Eigen to Torch
https://discuss.pytorch.org/t/data-transfer-between-libtorch-c-and-eigen/54156/6

## Windows

On Windows I could only build the project in Visual Studio cmake generator, couldn't make it work with MinGW/Unix
Makefiles. OpenCV also needs to be built with Visual Studio cmake generator. Also, if needed, compile OpenCV as both
`Debug` and `Release`, and download both Libtorch `Debug` and `Release`. Setup `%PATH%` accordingly.  
