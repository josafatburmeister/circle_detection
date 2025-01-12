#include <cmath>

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

namespace CircleDetection {

template <typename scalar_T>
scalar_T score_fn_scalar(scalar_T scaled_residual) {
  const scalar_T SQRT_2_PI = 2.5066282746310002;
  return exp(-(scaled_residual * scaled_residual) / 2) / SQRT_2_PI;
}

template <typename scalar_T>
scalar_T loss_fn_scalar(scalar_T scaled_residual) {
  return -score_fn_scalar<scalar_T>(scaled_residual);
}

template <typename scalar_T>
scalar_T loss_fn_derivative_1_scalar(scalar_T scaled_residual) {
  return -loss_fn_scalar<scalar_T>(scaled_residual) * scaled_residual;
}

template <typename scalar_T>
scalar_T loss_fn_derivative_2_scalar(scalar_T scaled_residual) {
  return loss_fn_scalar<scalar_T>(scaled_residual) * (scaled_residual * scaled_residual - 1);
}
}  // namespace CircleDetection

#endif  // LOSS_FUNCTIONS_H
