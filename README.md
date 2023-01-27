# MNIST database Neural Network in C++ and Python

This project aims to demonstrate a simple Neural Network using only math libraries (`numpy` for Python and `Eigen` for C++). It is not intent to use in production code.  

Download MNIST database from http://yann.lecun.com/exdb/mnist/  
Files: train images, train labels, test images, test labels  
Files are in `idx` format  

C++ code needs `Eigen` library as math library (matrices algebra) https://eigen.tuxfamily.org/. I used `Eigen 3.4.0`.  

C++ code needs `OpenCV` libraries to save images on disk https://docs.opencv.org/4.6.0/d7/d9f/tutorial_linux_install.html. I used `OpenCV 4.6.0`.  

C++ code needs `SimpleIni` include files to parse and use `config.ini` https://github.com/brofield/simpleini/. I used `SimpleInit 4.19`.

C++ code to read MNIST database was adapted from https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c  

Python code uses `numpy` as math library (matrices algebra) https://numpy.org/. I used `numpy 9.4.0`.  

Python code uses `pillow` to save images to disk https://pillow.readthedocs.io/en/stable/. I used `numpy 1.24.1`.  

```
pip install pillow==9.4.0
pip install numpy==1.24.1
```

Python code to read MNIST database was adapted from https://jamesmccaffrey.wordpress.com/2020/05/05/converting-raw-mnist-binary-files-to-text-files/  

Python code for the Neural Network was adapted from a tutorial from Samson Zhang [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://www.youtube.com/watch?v=w8yWXqWQYmU)  

C++ code for the Neural Network is an adaptation from myself based on the Python code.  