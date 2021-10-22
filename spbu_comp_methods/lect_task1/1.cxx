#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
 
using namespace std;
using namespace Eigen;
//using Eigen::MatrixXd;
//using Eigen::Matrix2d;

/* Eigen values example
  EigenSolver<MatrixXf> es;
  MatrixXf A = MatrixXf::Random(4,4);
  es.compute(A);
  cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << endl;
  es.compute(A + MatrixXf::Identity(4,4), false); // re-use es to compute eigenvalues of A+I
  cout << "The eigenvalues of A+I are: " << es.eigenvalues().transpose() << endl;
*/

/* Eigen vectors example
MatrixXd ones = MatrixXd::Ones(3,3);
EigenSolver<MatrixXd> es(ones);
cout << "The first eigenvector of the 3x3 matrix of ones is:"
     << endl << es.eigenvectors().col(0) << endl;
*/

Matrix2d identity(int n) {
  return MatrixXf::Identity(n,n);
  /*
  Matrix2d res(n, n);
  for (int i = 0; i < n; ++i) {
    res(i, i) = 1;
  }
  return res;
  */
}

Matrix2d inv(const Matrix2d& m) {
  return m.inverse();
}

double norm(const Matrix2d& m) {
  return m.norm();
}

double cond(Matrix2d m) {
  return 0;
}

double diag(Matrix2d m) {
  return 0;
}

vector<Matrix2d> eig(Matrix2d m) {
  EigenSolver<Matrix2d> es(m);
  vector<Matrix2d> res;
  auto vectors = es.eigenvectors();
  for (int i = 0; i < vectors.rows(); ++i) {
    res.push_back(vectors.col(i));
  }
  return res;
}

Matrix2d gen_vandermonde(int n) {
  //auto result = np.zeros(shape=(n,n));
  Matrix2d result(n, n);
  for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k)
          result(i,k) = pow((n-i+1), (-3*k/2));
  return result;
}

Matrix2d newton_method(Matrix2d X, double epsilon) {
  //auto x_k = identity(X.shape[0]);
  auto x_k = identity(X.rows());
  auto x_k1 = 0.5*(x_k + x_k.inverse() * X);
  while (norm(x_k1-x_k) > epsilon) {
    x_k = x_k1;
    //x_k1 = 0.5 * (x_k + x_k.inverse() * X);
  }
  return x_k1;
}

Matrix2d eig_method(Matrix2d X) {
  auto V = eig(X)[1];
  auto sigma = diag(MatrixBase::power(eig(X)[0], (0.5)));
  return V * sigma * inv(V);
}
 
int main() {
  vector<vector<double>> conds;
  for (int i = 2; i < 11; ++i) {
    auto X = gen_vandermonde(i);
    auto B1 = eig_method(X);
    if (i >= 2 || i <= 8) {
      auto B2 = newton_method(X,1e-3*pow(i, 2));
    }
    else {
      auto B2 = newton_method(X,1e-2*pow(i, 2));
    }
    conds.push_back(vector<double>{i,cond(X),cond(B1),cond(B2),norm(B2*B2-X)});
  }
  // Обобщенная матрица Вандермонда второго порядка:
  auto X = gen_vandermonde(2);
  cout << X << endl;
  // Корень из обобщенной матрицы Вандермонда второго порядка:
  cout << eig_method(X) << endl;
  // Обобщенная матрица Вандермонда третьего порядка:
  X = gen_vandermonde(3)
  cout << X << endl;
  // Корень из обобщенной матрицы Вандермонда третьего порядка:
  cout << eig_method(X) << endl;
  //cout << tabulate(conds,headers=['n','cond1','cond2','cond3','norm((X - NewtonMeth^2) << endl;'],
  //             tablefmt='github',numalign="right"))
}
