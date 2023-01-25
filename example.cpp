#include "include/eigen3/Eigen/Dense"
#include "include/eigen3/unsupported/Eigen/MatrixFunctions"
#include <iostream>

using namespace Eigen;

int main()
{
  const double pi = std::acos(-1.0);

  VectorXf p(3);
  p << 1, 2, 3;
  VectorXf y(3);
  y << 0, 2, 3;
  Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> e = p.cwiseEqual(y);
  VectorXi e2 = e.unaryExpr([](const bool x)
                        { return x ? 1 : 0; });
  std::cout << e2 << std::endl;
  std::cout << e2.sum() << std::endl;
}