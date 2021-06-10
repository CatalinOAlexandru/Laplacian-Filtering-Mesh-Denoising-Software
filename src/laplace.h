#ifndef laplace
#define laplace
#include <random>
#include <iostream>
#include <vector>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/fit_plane.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/SVD>


Eigen::VectorXd getH(Eigen::MatrixXd V,Eigen::MatrixXi F,bool cotan);
Eigen::SparseMatrix<double> getM(Eigen::MatrixXd V,Eigen::MatrixXi F);
double getArea(Eigen::RowVector3d a,Eigen::RowVector3d b,Eigen::RowVector3d c);
Eigen::SparseMatrix<double> invSparseMat(Eigen::SparseMatrix<double> M);
Eigen::SparseMatrix<double> uniformLaplace(Eigen::MatrixXd V,Eigen::MatrixXi F);
Eigen::VectorXd estimateGaussCurv(Eigen::MatrixXd V,Eigen::MatrixXi F);
Eigen::SparseMatrix<double> nonUniLaplace(Eigen::MatrixXd V,Eigen::MatrixXi F);
double getAngle(Eigen::RowVector3d a,Eigen::RowVector3d b,Eigen::RowVector3d c);
Eigen::MatrixXd meshReconstruction(Eigen::MatrixXd V,Eigen::MatrixXi F,int k);
Eigen::MatrixXd explicitSmoothing(Eigen::MatrixXd V,Eigen::MatrixXi F,double lambda,int numIters);
Eigen::MatrixXd implicitSmoothing(Eigen::MatrixXd V,Eigen::MatrixXi F,double lambda,int numIters);
Eigen::MatrixXd addNoise(Eigen::MatrixXd V, double std);
#endif


