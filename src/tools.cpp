#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::Block;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd(4);
  rmse.fill(0);
  if (estimations.size() != ground_truth.size()) {
    cout << "Cannot calculate RMSE" << endl;
    return rmse;
  }
  for (int i=0; i<estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse = rmse + residual;
  }
  rmse = (rmse/estimations.size()).array().sqrt();
  return rmse;
}

float Tools::NormalizeAngle(float angle) {
  return atan2(sin(angle), cos(angle));
}
