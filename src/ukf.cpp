#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_= false;

  time_us_ = 0;

  n_x_ = 5;

  n_aug_= 7;

  weights_ = VectorXd(2*n_aug_+1);

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  lambda_ = 3 - n_aug_;

  NIS_radar_ = 0.0;

  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (is_initialized_) {
    float dt = (meas_package.timestamp_ - time_us_)/1000000.0;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    } else {
      UpdateRadar(meas_package);
    }
  } else {
    time_us_ = meas_package.timestamp_;
    P_ = MatrixXd::Identity(5,5);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      x_ << px, py, 0, 0, 0;
    } else {
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];
      x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
    }
    is_initialized_ = true;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  Tools tools;

  // Augment
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0.0, 0.0;
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug << P_, MatrixXd::Zero(n_x_, 2),
           MatrixXd::Zero(2, n_x_), Q;

  // Generate sigma points
  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd sig_vec = sqrt(lambda_+n_aug_) * A;
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = sig_vec.colwise() + x_aug;
  Xsig_aug.block(0, 1+n_aug_, n_aug_, n_aug_) = (-sig_vec).colwise() + x_aug;

  // Predict sigma points
  for (int i=0; i<Xsig_aug.cols(); i++) {
    VectorXd pt = Xsig_aug.col(i);
    VectorXd x = pt.head(n_x_), noise = pt.tail(2);
    float px = x[0], py = x[1], v = x[2], yaw = x[3], yaw_rate = x[4];
    float noise_a = noise[0], noise_yaw = noise[1];
    VectorXd x_inc(5), x_err(5);
    if (fabs(yaw_rate) < 0.001) {
      x_inc << v*cos(yaw)*delta_t,
               v*sin(yaw)*delta_t,
               0,
               yaw_rate*delta_t,
               0;
    } else {
      x_inc << (v/yaw_rate)*(sin(yaw + yaw_rate*delta_t) - sin(yaw)),
               (v/yaw_rate)*(-cos(yaw + yaw_rate*delta_t) + cos(yaw)),
               0,
               yaw_rate*delta_t,
               0;
    }
    x_err << (1/2) * (delta_t*delta_t) * (cos(yaw)*noise_a),
             (1/2) * (delta_t*delta_t) * (sin(yaw)*noise_a),
             delta_t * noise_a,
             (1/2) * (delta_t*delta_t) * noise_yaw,
             delta_t*noise_yaw;
    Xsig_pred_.col(i) = x + x_inc + x_err;
  }

  // Predict mean and covariance
  double w1 = lambda_/(lambda_+n_aug_), w2 = 1.0/(2.0*(lambda_+n_aug_));
  weights_ << w1, VectorXd::Constant(2*n_aug_, w2);
  x_ = (Xsig_pred_.array().rowwise() * weights_.transpose().array()).rowwise().sum();
  MatrixXd Xdiff = Xsig_pred_.colwise() - x_;

  // Normalize
  for (int i = 0; i < Xdiff.cols(); i++) {
    VectorXd pt = Xdiff.col(i);
    pt[3] = tools.NormalizeAngle(pt[3]);
    Xdiff.col(i) = pt;
  }

  P_ = (Xdiff.array().rowwise() * weights_.transpose().array()).matrix() * Xdiff.transpose();
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd H = MatrixXd(2, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  Tools tools;
  VectorXd y = z - H*x_;
  MatrixXd PHt = P_*H.transpose();
  MatrixXd Sinv = (H*PHt + R).inverse();
  MatrixXd K = PHt*Sinv;
  const long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  x_ = x_ + K*y;
  P_ = (I - K*H) * P_;

  NIS_laser_ = y.transpose() * Sinv * y;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  Tools tools;

  // Transform sigma pts to measurement space
  MatrixXd Zsig = MatrixXd(3, 2*n_aug_+1);
  for (int i=0; i<Xsig_pred_.cols(); i++) {
    VectorXd pt = Xsig_pred_.col(i);
    float px = pt[0], py = pt[1], v = pt[2], yaw = pt[3], yaw_rate = pt[4];
    float rho = sqrt(px*px + py*py), phi = atan2(py, px);
    if (fabs(rho) < 0.001) rho = 0.001 * (rho/rho);
    float rho_dot = (px*cos(yaw)*v + py*sin(yaw)*v)/rho;
    Zsig.col(i) << rho, phi, rho_dot;
  }

  // Predict measurement z at k+1
  VectorXd z_pred = (Zsig.array().rowwise() * weights_.transpose().array()).rowwise().sum();
  z_pred[1] = tools.NormalizeAngle(z_pred[1]);
  MatrixXd Zdiff = Zsig.colwise() - z_pred;

  // Normalize
  for (int i = 0; i < Zdiff.cols(); i++) {
    VectorXd pt = Zdiff.col(i);
    pt[1] = tools.NormalizeAngle(pt[1]);
    Zdiff.col(i) = pt;
  }

  MatrixXd Zdiff_weighted = Zdiff.array().rowwise() * weights_.transpose().array();
  MatrixXd R = MatrixXd(3, 3);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  MatrixXd S = Zdiff_weighted * Zdiff.transpose() + R;

  // Actual measurement z at k+1
  VectorXd z = meas_package.raw_measurements_;

  // Update state and covariance with measurement
  MatrixXd Xdiff = Xsig_pred_.colwise() - x_;
  // Normalize
  for (int i = 0; i < Xdiff.cols(); i++) {
    VectorXd pt = Xdiff.col(i);
    pt[3] = tools.NormalizeAngle(pt[3]);
    Xdiff.col(i) = pt;
  }

  MatrixXd Xdiff_weighted = Xdiff.array().rowwise() * weights_.transpose().array();
  MatrixXd Tc = Xdiff_weighted * Zdiff.transpose();

  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc * Sinv;
  VectorXd y = (z - z_pred);
  y[1] = tools.NormalizeAngle(y[1]);
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = y.transpose() * Sinv * y;
}
