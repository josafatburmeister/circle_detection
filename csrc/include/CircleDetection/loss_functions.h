#include <cmath>

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

namespace CircleDetection {

double score_fn_scalar(double scaled_residual) {
  const double SQRT_2_PI = 2.5066282746310002;
  return exp(-(scaled_residual * scaled_residual) / 2) / SQRT_2_PI;
}

double loss_fn_scalar(double scaled_residual) { return -score_fn_scalar(scaled_residual); }

double loss_fn_derivative_1_scalar(double scaled_residual) {
  return -loss_fn_scalar(scaled_residual) * scaled_residual;
}

double loss_fn_derivative_2_scalar(double scaled_residual) {
  return loss_fn_scalar(scaled_residual) * (scaled_residual * scaled_residual - 1);
}
}  // namespace CircleDetection

#endif  // LOSS_FUNCTIONS_H
