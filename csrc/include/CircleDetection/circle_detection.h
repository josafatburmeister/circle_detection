#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

namespace {

using namespace Eigen;
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using ArrayXl = Eigen::Array<long, Eigen::Dynamic, 1>;

double loss_fn_scalar(double scaled_residual) {
  const double SQRT_2_PI = 2.5066282746310002;
  return -exp(-(scaled_residual * scaled_residual) / 2) / SQRT_2_PI;
}

double loss_fn_derivative_1_scalar(double scaled_residual) {
  return -loss_fn_scalar(scaled_residual) * scaled_residual;
}

double loss_fn_derivative_2_scalar(double scaled_residual) {
  return loss_fn_scalar(scaled_residual) * (scaled_residual * scaled_residual - 1);
}
}  // namespace

namespace CircleDetection {
std::tuple<std::vector<ArrayX3d>, std::vector<ArrayXd>> detect_circles(
    ArrayX2d xy, std::vector<std::vector<int64_t>> batch_indices, double bandwidth, ArrayXd min_start_x,
    ArrayXd max_start_x, int n_start_x, ArrayXd min_start_y, ArrayXd max_start_y, int n_start_y,
    ArrayXd min_start_radius, ArrayXd max_start_radius, int n_start_radius, ArrayXd break_min_x, ArrayXd break_max_x,
    ArrayXd break_min_y, ArrayXd break_max_y, ArrayXd break_min_radius, ArrayXd break_max_radius,
    double break_min_change = 1e-5, int max_iterations = 1000, double acceleration_factor = 1.6,
    double armijo_attenuation_factor = 0.7, double armijo_min_decrease_percentage = 0.5, double min_step_size = 1e-20,
    double min_fitting_score = 1e-6) {
  auto n_batches = batch_indices.size() > 0 ? batch_indices.size() : 1;

  ArrayXd start_radii(n_batches * n_start_radius);
  ArrayXd start_centers_x(n_batches * n_start_x);
  ArrayXd start_centers_y(n_batches * n_start_y);

#pragma omp parallel for
  for (int64_t i = 0; i < n_batches; ++i) {
    start_radii.segment(i * n_start_radius, n_start_radius) =
        ArrayXd::LinSpaced(n_start_radius, min_start_radius(i), max_start_radius(i));
    start_centers_x.segment(i * n_start_x, n_start_x) = ArrayXd::LinSpaced(n_start_x, min_start_x(i), max_start_x(i));
    start_centers_y.segment(i * n_start_y, n_start_y) = ArrayXd::LinSpaced(n_start_y, min_start_y(i), max_start_y(i));
  }

  ArrayX3d fitted_circles = ArrayX3d::Constant(n_batches * n_start_radius * n_start_x * n_start_y, 3, -1);
  ArrayXb fitting_converged = ArrayXb::Zero(n_batches * n_start_radius * n_start_x * n_start_y);
  ArrayXd fitting_losses = ArrayXd::Constant(n_batches * n_start_radius * n_start_x * n_start_y, 0);

#pragma omp parallel
#pragma omp single
  for (int64_t idx_batch = 0; idx_batch < n_batches; ++idx_batch) {
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
          if (batch_indices.size() > 0 && batch_indices[idx_batch].size() < 3) {
            continue;
          }

#pragma omp task
          {
            ArrayX2d current_xy;
            if (n_batches > 1) {
              current_xy = xy(batch_indices[idx_batch], Eigen::all);
            } else {
              current_xy = xy;
            }

            auto start_radius = start_radii[idx_batch * n_start_radius + idx_radius];
            auto radius = start_radius;

            auto start_center_x = start_centers_x[idx_batch * n_start_x + idx_x];
            auto start_center_y = start_centers_y[idx_batch * n_start_y + idx_y];
            RowVector2d center(start_center_x, start_center_y);

            double fitting_loss = 0;
            double fitting_score = 0;
            bool diverged = false;

            for (int iteration = 0; iteration < max_iterations; ++iteration) {
              ArrayXd squared_dists_to_center =
                  (current_xy.matrix().rowwise() - center).rowwise().squaredNorm().array();
              ArrayXd dists_to_center = squared_dists_to_center.array().sqrt();
              ArrayXd scaled_residuals = (dists_to_center - radius) / bandwidth;
              fitting_loss = scaled_residuals.unaryExpr(&loss_fn_scalar).sum();

              // first derivative of the outer term of the loss function
              ArrayXd outer_derivative_1 = scaled_residuals.unaryExpr(&loss_fn_derivative_1_scalar);

              // second derivative of the outer term of the loss function
              ArrayXd outer_derivative_2 = scaled_residuals.unaryExpr(&loss_fn_derivative_2_scalar);

              // first derivative of the inner term of the loss function
              // this array stores the derivatives dx and dy in different columns
              ArrayX2d inner_derivative_1_x = (-1 / (bandwidth * dists_to_center)).replicate(1, 2) *
                                              (current_xy.matrix().rowwise() - center).array();
              double inner_derivative_1_r = -1 / bandwidth;

              // second derivative of the inner term of the loss function
              // this array stores the derivatives dxdx and dydy in different columns
              ArrayX2d inner_derivative_2_x_x = 1 / bandwidth *
                                                (-1 / (squared_dists_to_center * dists_to_center).replicate(1, 2) *
                                                     (current_xy.matrix().rowwise() - center).array().square() +
                                                 1 / dists_to_center.replicate(1, 2));
              // this array stores the derivatives dxdy and dydx in one column (both are identical)
              ArrayXd inner_derivative_2_x_y = -1 / bandwidth * 1 / (squared_dists_to_center * dists_to_center) *
                                               (current_xy.col(0) - center[0]) * (current_xy.col(1) - center[1]);

              // first derivatives of the entire loss function with respect to the circle parameters
              RowVector2d derivative_xy =
                  (outer_derivative_1.replicate(1, 2) * inner_derivative_1_x).matrix().colwise().sum();
              double derivative_r = (outer_derivative_1 * inner_derivative_1_r).matrix().sum();
              Vector3d gradient(derivative_xy[0], derivative_xy[1], derivative_r);

              // second derivatives of the entire loss function with respect to the circle parameters
              double derivative_x_x = ((outer_derivative_2 * inner_derivative_1_x.col(0).square()) +
                                       (outer_derivative_1 * inner_derivative_2_x_x.col(0)))
                                          .matrix()
                                          .sum();
              double derivative_x_y =
                  ((outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_x.col(0)) +
                   (outer_derivative_1 * inner_derivative_2_x_y))
                      .matrix()
                      .sum();
              double derivative_x_r =
                  (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(0)).matrix().sum();

              double derivative_y_x =
                  ((outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_x.col(1)) +
                   (outer_derivative_1 * inner_derivative_2_x_y))
                      .matrix()
                      .sum();
              double derivative_y_y = ((outer_derivative_2 * inner_derivative_1_x.col(1).square()) +
                                       (outer_derivative_1 * inner_derivative_2_x_x.col(1)))
                                          .matrix()
                                          .sum();
              double derivative_y_r =
                  (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(1)).matrix().sum();

              double derivative_r_x =
                  (outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_r).matrix().sum();
              double derivative_r_y =
                  (outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_r).matrix().sum();
              double derivative_r_r = (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_r).matrix().sum();

              Matrix3d hessian(3, 3);
              hessian << derivative_x_x, derivative_x_y, derivative_x_r, derivative_y_x, derivative_y_y, derivative_y_r,
                  derivative_r_x, derivative_r_y, derivative_r_r;

              double determinant_hessian = hessian.determinant();

              double determinant_hessian_submatrix = derivative_x_x * derivative_y_y - derivative_x_y * derivative_y_x;

              double step_size = 1.0;
              ArrayXd step_direction(3);
              if ((determinant_hessian > 0) && (determinant_hessian_submatrix > 0)) {
                step_direction = -1 * (hessian.inverse() * gradient).array();
              } else {
                step_direction = -1 * gradient;

                // step size acceleration
                double next_step_size = 1.0;
                auto next_center = center + (next_step_size * step_direction.head(2)).matrix().transpose();
                auto next_radius = radius + (next_step_size * step_direction[2]);
                ArrayXd next_scaled_residuals =
                    ((current_xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) / bandwidth;
                auto next_loss = next_scaled_residuals.unaryExpr(&loss_fn_scalar).sum();
                auto previous_loss = fitting_loss;

                while (next_loss < previous_loss) {
                  step_size = next_step_size;
                  fitting_score = -1 * next_loss;
                  previous_loss = next_loss;
                  next_step_size *= acceleration_factor;

                  auto next_center = center + (next_step_size * step_direction.head(2)).matrix().transpose();
                  auto next_radius = radius + (next_step_size * step_direction[2]);
                  ArrayXd next_scaled_residuals =
                      ((current_xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) /
                      bandwidth;
                  next_loss = next_scaled_residuals.unaryExpr(&loss_fn_scalar).sum();
                }
              }

              // step size attenuation according to Armijo's rule
              // if acceleration was successfull, the attenuation is skipped
              // if acceleration was not successfull, the stpe size is still 1
              if (step_size == 1) {
                // to avoid initializing all variables of the while loop before, actual_loss_diff is set to 1
                // and expected_loss_diff to 0 so that the loop is executed at least once and the variables are properly
                // initialized in the first iteration of the loop
                double actual_loss_diff = 1.0;
                double expected_loss_diff = 0.0;
                step_size = 1 / armijo_attenuation_factor;

                while (actual_loss_diff > expected_loss_diff && step_size > min_step_size) {
                  step_size *= armijo_attenuation_factor;

                  auto next_center = center + (step_size * step_direction.head(2)).matrix().transpose();
                  auto next_radius = radius + (step_size * step_direction[2]);
                  ArrayXd next_scaled_residuals =
                      ((current_xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) /
                      bandwidth;
                  auto next_loss = next_scaled_residuals.unaryExpr(&loss_fn_scalar).sum();
                  fitting_score = -1 * next_loss;

                  actual_loss_diff = next_loss - fitting_loss;
                  expected_loss_diff =
                      armijo_min_decrease_percentage * step_size * (gradient.transpose() * step_direction.matrix())[0];
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

              if ((abs(radius_update) < break_min_change) && (abs(center_update[0]) < break_min_change) &&
                  (abs(center_update[1]) < break_min_change)) {
                break;
              }
            }

            if (!diverged && fitting_score >= min_fitting_score && std::isfinite(center[0]) &&
                std::isfinite(center[1]) && std::isfinite(radius) && radius > 0) {
              int64_t idx = n_start_radius * (n_start_y * (idx_batch * n_start_x + idx_x) + idx_y) + idx_radius;
              fitted_circles(idx, 0) = center[0];
              fitted_circles(idx, 1) = center[1];
              fitted_circles(idx, 2) = radius;
              fitting_converged(idx) = true;
              fitting_losses(idx) = -1 * fitting_score;
            }
          }
        }
      }
    }
  }

#pragma omp taskwait

  std::vector<ArrayX3d> converged_circles;
  std::vector<ArrayXd> converged_fitting_losses;

  for (int64_t i = 0; i < n_batches; ++i) {
    std::vector<int64_t> converged_indices{};
    int64_t n_circles = n_start_x * n_start_y * n_start_radius;
    for (int64_t j = 0; j < n_circles; ++j) {
      auto idx = i * n_circles + j;
      if (fitting_converged[idx]) {
        converged_indices.push_back(idx);
      }
    }
    converged_circles.push_back(fitted_circles(converged_indices, Eigen::all));
    converged_fitting_losses.push_back(fitting_losses(converged_indices));
  }
  return std::make_tuple(converged_circles, converged_fitting_losses);
}

}  // namespace CircleDetection
