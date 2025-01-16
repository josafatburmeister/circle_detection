#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

#include "loss_functions.h"
#include "operations.h"
#include "type_aliases.h"

#ifndef CIRCLE_DETECTION_M_ESTIMATOR_H
#define CIRCLE_DETECTION_M_ESTIMATOR_H

namespace CircleDetection {

template <typename scalar_T>
std::tuple<ArrayX3<scalar_T>, ArrayX<scalar_T>, ArrayXl> detect_circles_m_estimator(
    RefArrayX2<scalar_T> xy,
    RefArrayXl batch_lengths,
    scalar_T bandwidth,
    RefArrayX<scalar_T> min_start_x,
    RefArrayX<scalar_T> max_start_x,
    int n_start_x,
    RefArrayX<scalar_T> min_start_y,
    RefArrayX<scalar_T> max_start_y,
    int n_start_y,
    RefArrayX<scalar_T> min_start_radius,
    RefArrayX<scalar_T> max_start_radius,
    int n_start_radius,
    RefArrayX<scalar_T> break_min_x,
    RefArrayX<scalar_T> break_max_x,
    RefArrayX<scalar_T> break_min_y,
    RefArrayX<scalar_T> break_max_y,
    RefArrayX<scalar_T> break_min_radius,
    RefArrayX<scalar_T> break_max_radius,
    scalar_T break_min_change = 1e-5,
    int max_iterations = 1000,
    scalar_T acceleration_factor = 1.6,
    scalar_T armijo_attenuation_factor = 0.7,
    scalar_T armijo_min_decrease_percentage = 0.5,
    scalar_T min_step_size = 1e-20,
    scalar_T min_fitting_score = 1,
    int num_workers = 1) {
  if (xy.rows() != batch_lengths.sum()) {
    throw std::invalid_argument("The number of points must be equal to the sum of batch_lengths");
  }

  if (min_start_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of min_start_x must be equal to the batch size.");
  }
  if (max_start_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of max_start_x must be equal to the batch size.");
  }
  if (break_min_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_min_x must be equal to the batch size.");
  }
  if (break_max_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_max_x must be equal to the batch size.");
  }

  if (min_start_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of min_start_y must be equal to the batch size.");
  }
  if (max_start_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of max_start_y must be equal to the batch size.");
  }
  if (break_min_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_min_y must be equal to the batch size.");
  }
  if (break_max_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_max_y must be equal to the batch size.");
  }

  if (min_start_radius.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of min_start_radius must be equal to the batch size.");
  }
  if (max_start_radius.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of max_start_radius must be equal to the batch size.");
  }
  if (break_min_radius.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_min_radius must be equal to the batch size.");
  }
  if (break_max_radius.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_max_radius must be equal to the batch size.");
  }

  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  auto num_batches = batch_lengths.rows();

  ArrayXl batch_starts(num_batches);
  ArrayX2<scalar_T> offsets(num_batches, 2);

  int64_t batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts(batch_idx) = batch_start;
    batch_start += batch_lengths(batch_idx);
  }

  std::vector<ArrayX2<scalar_T>> xy_per_batch(num_batches);

  // to improve the numerical stability the data are shifted before the circle detection
  // after circle detection, the inverse of the normalization is applied to the circle parameters
#pragma omp parallel for num_threads(num_workers)
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    if (batch_lengths(batch_idx) > 0) {
      ArrayX2<scalar_T> current_xy = xy(Eigen::seqN(batch_starts(batch_idx), batch_lengths(batch_idx)), Eigen::all);
      ArrayX<scalar_T> offset = current_xy.colwise().mean();
      offsets(batch_idx, Eigen::all) = offset;

      xy_per_batch[batch_idx] = current_xy.rowwise() - offset.transpose();
      min_start_x(batch_idx) = (min_start_x(batch_idx) - offset(0));
      max_start_x(batch_idx) = (max_start_x(batch_idx) - offset(0));
      min_start_y(batch_idx) = (min_start_y(batch_idx) - offset(1));
      max_start_y(batch_idx) = (max_start_y(batch_idx) - offset(1));
      break_min_x(batch_idx) = (break_min_x(batch_idx) - offset(0));
      break_max_x(batch_idx) = (break_max_x(batch_idx) - offset(0));
      break_min_y(batch_idx) = (break_min_y(batch_idx) - offset(1));
      break_max_y(batch_idx) = (break_max_y(batch_idx) - offset(1));
    }
  }

  ArrayX<scalar_T> start_radii(num_batches * n_start_radius);
  ArrayX<scalar_T> start_centers_x(num_batches * n_start_x);
  ArrayX<scalar_T> start_centers_y(num_batches * n_start_y);
  scalar_T eps = 0.01 * min_step_size;

#pragma omp parallel for num_threads(num_workers)
  for (int64_t i = 0; i < num_batches; ++i) {
    scalar_T offset_radius = (max_start_radius(i) - min_start_radius(i)) / static_cast<scalar_T>(n_start_radius + 1);
    start_radii(Eigen::seqN(i * n_start_radius, n_start_radius)) = ArrayX<scalar_T>::LinSpaced(
        n_start_radius, min_start_radius(i) + offset_radius, max_start_radius(i) - offset_radius);
    scalar_T offset_x = (max_start_x(i) - min_start_x(i)) / static_cast<scalar_T>(n_start_x + 1);
    start_centers_x(Eigen::seqN(i * n_start_x, n_start_x)) =
        ArrayX<scalar_T>::LinSpaced(n_start_x, min_start_x(i) + offset_x, max_start_x(i) - offset_x);
    scalar_T offset_y = (max_start_y(i) - min_start_y(i)) / static_cast<scalar_T>(n_start_y + 1);
    start_centers_y(Eigen::seqN(i * n_start_y, n_start_y)) =
        ArrayX<scalar_T>::LinSpaced(n_start_y, min_start_y(i) + offset_y, max_start_y(i) - offset_y);
  }

  ArrayX3<scalar_T> fitted_circles =
      ArrayX3<scalar_T>::Constant(num_batches * n_start_radius * n_start_x * n_start_y, 3, -1);
  ArrayXb fitting_converged = ArrayXb::Zero(num_batches * n_start_radius * n_start_x * n_start_y);
  ArrayX<scalar_T> fitting_scores = ArrayX<scalar_T>::Constant(num_batches * n_start_radius * n_start_x * n_start_y, 0);

#pragma omp parallel num_threads(num_workers)
#pragma omp single
  for (int64_t idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
    for (int64_t idx_x = 0; idx_x < n_start_x; ++idx_x) {
      for (int64_t idx_y = 0; idx_y < n_start_y; ++idx_y) {
        for (int64_t idx_radius = 0; idx_radius < n_start_radius; ++idx_radius) {
          if (min_start_x(idx_batch) == max_start_x(idx_batch) && idx_x > 0) {
            continue;
          }
          if (min_start_y(idx_batch) == max_start_y(idx_batch) && idx_y > 0) {
            continue;
          }
          if (min_start_radius(idx_batch) == max_start_radius(idx_batch) && idx_radius > 0) {
            continue;
          }
          if (batch_lengths(idx_batch) < 3) {
            continue;
          }

#pragma omp task
          {
            scalar_T start_radius = start_radii[idx_batch * n_start_radius + idx_radius];
            scalar_T radius = start_radius;

            scalar_T start_center_x = start_centers_x[idx_batch * n_start_x + idx_x];
            scalar_T start_center_y = start_centers_y[idx_batch * n_start_y + idx_y];
            RowVector2<scalar_T> center(start_center_x, start_center_y);

            scalar_T fitting_loss = 0;
            scalar_T fitting_score = 0;
            bool diverged = false;

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
              ArrayX<scalar_T> squared_dists_to_center =
                  (xy_per_batch[idx_batch].matrix().rowwise() - center).rowwise().squaredNorm().array();
              ArrayX<scalar_T> dists_to_center = squared_dists_to_center.sqrt();
              ArrayX<scalar_T> scaled_residuals = (dists_to_center - radius) / bandwidth;
              fitting_loss = scaled_residuals.unaryExpr(&CircleDetection::loss_fn_scalar<scalar_T>).mean();

              // first derivative of the outer term of the loss function
              ArrayX<scalar_T> outer_derivative_1 =
                  scaled_residuals.unaryExpr(&CircleDetection::loss_fn_derivative_1_scalar<scalar_T>);

              // second derivative of the outer term of the loss function
              ArrayX<scalar_T> outer_derivative_2 =
                  scaled_residuals.unaryExpr(&CircleDetection::loss_fn_derivative_2_scalar<scalar_T>);

              // first derivative of the inner term of the loss function
              // this array stores the derivatives dx and dy in different columns
              ArrayX2<scalar_T> inner_derivative_1_x = (-1 / (bandwidth * dists_to_center)).replicate(1, 2) *
                                                       (xy_per_batch[idx_batch].matrix().rowwise() - center).array();
              scalar_T inner_derivative_1_r = -1 / bandwidth;

              // second derivative of the inner term of the loss function
              // this array stores the derivatives dxdx and dydy in different columns
              ArrayX2<scalar_T> inner_derivative_2_x_x =
                  1 / bandwidth *
                  (-1 / (squared_dists_to_center * dists_to_center).replicate(1, 2) *
                       (xy_per_batch[idx_batch].matrix().rowwise() - center).array().square() +
                   1 / dists_to_center.replicate(1, 2));
              // this array stores the derivatives dxdy and dydx in one column (both are identical)
              ArrayX<scalar_T> inner_derivative_2_x_y =
                  -1 / bandwidth * 1 / (squared_dists_to_center * dists_to_center) *
                  (xy_per_batch[idx_batch].col(0) - center[0]) * (xy_per_batch[idx_batch].col(1) - center[1]);

              // first derivatives of the entire loss function with respect to the circle parameters
              RowVector2<scalar_T> derivative_xy =
                  (outer_derivative_1.replicate(1, 2) * inner_derivative_1_x).matrix().colwise().mean();
              scalar_T derivative_r = (outer_derivative_1 * inner_derivative_1_r).matrix().mean();
              Vector3<scalar_T> gradient(derivative_xy[0], derivative_xy[1], derivative_r);

              // second derivatives of the entire loss function with respect to the circle parameters
              scalar_T derivative_x_x = ((outer_derivative_2 * inner_derivative_1_x.col(0).square()) +
                                         (outer_derivative_1 * inner_derivative_2_x_x.col(0)))
                                            .mean();
              scalar_T derivative_x_y =
                  ((outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_x.col(0)) +
                   (outer_derivative_1 * inner_derivative_2_x_y))
                      .mean();
              scalar_T derivative_x_r =
                  (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(0)).mean();

              scalar_T derivative_y_x =
                  ((outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_x.col(1)) +
                   (outer_derivative_1 * inner_derivative_2_x_y))
                      .matrix()
                      .mean();
              scalar_T derivative_y_y = ((outer_derivative_2 * inner_derivative_1_x.col(1).square()) +
                                         (outer_derivative_1 * inner_derivative_2_x_x.col(1)))
                                            .mean();
              scalar_T derivative_y_r =
                  (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(1)).mean();

              scalar_T derivative_r_x =
                  (outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_r).mean();
              scalar_T derivative_r_y =
                  (outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_r).mean();
              scalar_T derivative_r_r = (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_r).mean();

              MatrixX3<scalar_T> hessian(3, 3);
              hessian << derivative_x_x, derivative_x_y, derivative_x_r, derivative_y_x, derivative_y_y, derivative_y_r,
                  derivative_r_x, derivative_r_y, derivative_r_r;

              scalar_T determinant_hessian = hessian.determinant();

              scalar_T determinant_hessian_submatrix =
                  derivative_x_x * derivative_y_y - derivative_x_y * derivative_y_x;

              scalar_T step_size = 1.0;
              ArrayX<scalar_T> step_direction(3);
              if ((determinant_hessian > 0) && (determinant_hessian_submatrix > 0)) {
                step_direction = -1 * (hessian.inverse() * gradient).array();
              } else {
                step_direction = -1 * gradient;

                // step size acceleration
                scalar_T next_step_size = 1.0;
                auto next_center = center + (next_step_size * step_direction.head(2)).matrix().transpose();
                auto next_radius = radius + (next_step_size * step_direction[2]);
                ArrayX<scalar_T> next_scaled_residuals =
                    ((xy_per_batch[idx_batch].matrix().rowwise() - next_center).rowwise().norm().array() -
                     next_radius) /
                    bandwidth;
                auto next_loss = next_scaled_residuals.unaryExpr(&CircleDetection::loss_fn_scalar<scalar_T>).mean();
                auto previous_loss = fitting_loss;

                while (next_loss < previous_loss) {
                  step_size = next_step_size;
                  fitting_score = -1 * next_loss / bandwidth;
                  previous_loss = next_loss;
                  next_step_size *= acceleration_factor;

                  auto next_center = center + (next_step_size * step_direction.head(2)).matrix().transpose();
                  auto next_radius = radius + (next_step_size * step_direction[2]);
                  ArrayX<scalar_T> next_scaled_residuals =
                      ((xy_per_batch[idx_batch].matrix().rowwise() - next_center).rowwise().norm().array() -
                       next_radius) /
                      bandwidth;
                  next_loss = next_scaled_residuals.unaryExpr(&CircleDetection::loss_fn_scalar<scalar_T>).mean();
                }
              }

              // step size attenuation according to Armijo's rule
              // if acceleration was successfull, the attenuation is skipped
              // if acceleration was not successfull, the step size is still 1
              if (step_size == 1) {
                // to avoid initializing all variables of the while loop before, actual_loss_decrease is set to 0
                // and expected_loss_decrease to 1 so that the loop is executed at least once and the variables are
                // properly initialized in the first iteration of the loop
                scalar_T actual_loss_decrease = 0.0;
                scalar_T expected_loss_decrease = 1.0;
                step_size = 1 / armijo_attenuation_factor;

                while (expected_loss_decrease - actual_loss_decrease > eps && step_size > min_step_size) {
                  step_size *= armijo_attenuation_factor;

                  auto next_center = center + (step_size * step_direction.head(2)).matrix().transpose();
                  auto next_radius = radius + (step_size * step_direction[2]);
                  ArrayX<scalar_T> next_scaled_residuals =
                      ((xy_per_batch[idx_batch].matrix().rowwise() - next_center).rowwise().norm().array() -
                       next_radius) /
                      bandwidth;
                  auto next_loss = next_scaled_residuals.unaryExpr(&CircleDetection::loss_fn_scalar<scalar_T>).mean();
                  fitting_score = -1 * next_loss / bandwidth;

                  actual_loss_decrease = fitting_loss - next_loss;
                  expected_loss_decrease = -1 * armijo_min_decrease_percentage * step_size *
                                           (gradient.transpose() * step_direction.matrix())[0];
                }
              }

              auto center_update = (step_size * step_direction.head(2)).matrix().transpose();
              center = center + center_update;
              auto radius_update = step_size * step_direction[2];
              radius = radius + radius_update;

              if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(radius) ||
                  center[0] < break_min_x(idx_batch) || center[0] > break_max_x(idx_batch) ||
                  center[1] < break_min_y(idx_batch) || center[1] > break_max_y(idx_batch) ||
                  radius < break_min_radius(idx_batch) || radius > break_max_radius(idx_batch) || radius <= 0 ||
                  iteration == max_iterations - 1) {
                diverged = true;
                break;
              }

              if ((std::abs(radius_update) < break_min_change) && (std::abs(center_update[0]) < break_min_change) &&
                  (std::abs(center_update[1]) < break_min_change)) {
                break;
              }
            }

            if (!diverged && fitting_score >= min_fitting_score && std::isfinite(center[0]) &&
                std::isfinite(center[1]) && std::isfinite(radius) && radius > 0) {
              int64_t idx = n_start_radius * (n_start_y * (idx_batch * n_start_x + idx_x) + idx_y) + idx_radius;
              // revert the shifting
              fitted_circles(idx, 0) = center[0] + offsets(idx_batch, 0);
              fitted_circles(idx, 1) = center[1] + offsets(idx_batch, 1);
              fitted_circles(idx, 2) = radius;
              fitting_converged(idx) = true;
              fitting_scores(idx) = fitting_score;
            }
          }
        }
      }
    }
  }

#pragma omp taskwait

  ArrayXl batch_lengths_circles = ArrayXl::Constant(num_batches, 0);
  std::vector<int64_t> converged_indices;

  for (int64_t i = 0; i < fitted_circles.rows(); ++i) {
    if (fitting_converged[i]) {
      converged_indices.push_back(i);
      batch_lengths_circles(i / (n_start_x * n_start_y * n_start_radius)) += 1;
    }
  }

  return std::make_tuple(
      fitted_circles(converged_indices, Eigen::all), fitting_scores(converged_indices), batch_lengths_circles);
}

}  // namespace CircleDetection

#endif  // CIRCLE_DETECTION_M_ESTIMATOR_H
