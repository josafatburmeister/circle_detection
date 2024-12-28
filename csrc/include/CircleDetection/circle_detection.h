#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using ArrayXl = Eigen::Array<long, Eigen::Dynamic, 1>;

using namespace Eigen;

namespace {
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
std::tuple<ArrayX3d, ArrayXd> detect_circles(ArrayX2d xy, double bandwidth, double min_start_x, double max_start_x,
                                             int n_start_x, double min_start_y, double max_start_y, int n_start_y,
                                             double min_start_radius, double max_start_radius, int n_start_radius,
                                             double break_min_x, double break_max_x, double break_min_y,
                                             double break_max_y, double break_min_radius, double break_max_radius,
                                             double break_min_change = 1e-5, int max_iterations = 1000,
                                             double acceleration_factor = 1.6, double armijo_attenuation_factor = 0.7,
                                             double armijo_min_decrease_percentage = 0.5, double min_step_size = 1e-20,
                                             double min_fitting_score = 1e-6) {
  if (min_start_x == max_start_x) {
    n_start_x = 1;
  }

  if (min_start_y == max_start_y) {
    n_start_y = 1;
  }

  if (min_start_radius == max_start_radius) {
    n_start_radius = 1;
  }

  ArrayXd start_radii = ArrayXd::LinSpaced(n_start_radius, min_start_radius, max_start_radius);
  ArrayXd start_centers_x = ArrayXd::LinSpaced(n_start_x, min_start_x, max_start_x);
  ArrayXd start_centers_y = ArrayXd::LinSpaced(n_start_y, min_start_y, max_start_y);

  ArrayX3d fitted_circles = ArrayX3d::Constant(n_start_radius * n_start_x * n_start_y, 3, -1);
  ArrayXb fitting_converged = ArrayXb::Zero(n_start_radius * n_start_x * n_start_y);
  ArrayXd fitting_losses = ArrayXd::Constant(n_start_radius * n_start_x * n_start_y, 0);

#pragma omp parallel for
  for (int idx = 0; idx < n_start_radius * n_start_x * n_start_y; ++idx) {
    auto idx_1 = idx / (n_start_x * n_start_y);
    auto remainder = idx % (n_start_x * n_start_y);
    auto idx_2 = remainder / n_start_y;
    auto idx_3 = remainder % n_start_y;

    auto start_radius = start_radii[idx_1];
    auto radius = start_radius;

    auto start_center_x = start_centers_x[idx_2];
    auto start_center_y = start_centers_y[idx_3];
    RowVector2d center(start_center_x, start_center_y);

    double fitting_loss = 0;
    double fitting_score = 0;
    bool diverged = false;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      ArrayXd squared_dists_to_center = (xy.matrix().rowwise() - center).rowwise().squaredNorm().array();
      ArrayXd dists_to_center = squared_dists_to_center.array().sqrt();
      ArrayXd scaled_residuals = (dists_to_center - radius) / bandwidth;
      fitting_loss = scaled_residuals.unaryExpr(&loss_fn_scalar).sum();

      // first derivative of the outer term of the loss function
      ArrayXd outer_derivative_1 = scaled_residuals.unaryExpr(&loss_fn_derivative_1_scalar);

      // second derivative of the outer term of the loss function
      ArrayXd outer_derivative_2 = scaled_residuals.unaryExpr(&loss_fn_derivative_2_scalar);

      // first derivative of the inner term of the loss function
      // this array stores the derivatives dx and dy in different columns
      ArrayX2d inner_derivative_1_x =
          (-1 / (bandwidth * dists_to_center)).replicate(1, 2) * (xy.matrix().rowwise() - center).array();
      double inner_derivative_1_r = -1 / bandwidth;

      // second derivative of the inner term of the loss function
      // this array stores the derivatives dxdx and dydy in different columns
      ArrayX2d inner_derivative_2_x_x = 1 / bandwidth *
                                        (-1 / (squared_dists_to_center * dists_to_center).replicate(1, 2) *
                                             (xy.matrix().rowwise() - center).array().square() +
                                         1 / dists_to_center.replicate(1, 2));
      // this array stores the derivatives dxdy and dydx in one column (both are identical)
      ArrayXd inner_derivative_2_x_y = -1 / bandwidth * 1 / (squared_dists_to_center * dists_to_center) *
                                       (xy.col(0) - center[0]) * (xy.col(1) - center[1]);

      // first derivatives of the entire loss function with respect to the circle parameters
      RowVector2d derivative_xy = (outer_derivative_1.replicate(1, 2) * inner_derivative_1_x).matrix().colwise().sum();
      double derivative_r = (outer_derivative_1 * inner_derivative_1_r).matrix().sum();
      Vector3d gradient(derivative_xy[0], derivative_xy[1], derivative_r);

      // second derivatives of the entire loss function with respect to the circle parameters
      double derivative_x_x = ((outer_derivative_2 * inner_derivative_1_x.col(0).square()) +
                               (outer_derivative_1 * inner_derivative_2_x_x.col(0)))
                                  .matrix()
                                  .sum();
      double derivative_x_y = ((outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_x.col(0)) +
                               (outer_derivative_1 * inner_derivative_2_x_y))
                                  .matrix()
                                  .sum();
      double derivative_x_r = (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(0)).matrix().sum();

      double derivative_y_x = ((outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_x.col(1)) +
                               (outer_derivative_1 * inner_derivative_2_x_y))
                                  .matrix()
                                  .sum();
      double derivative_y_y = ((outer_derivative_2 * inner_derivative_1_x.col(1).square()) +
                               (outer_derivative_1 * inner_derivative_2_x_x.col(1)))
                                  .matrix()
                                  .sum();
      double derivative_y_r = (outer_derivative_2 * inner_derivative_1_r * inner_derivative_1_x.col(1)).matrix().sum();

      double derivative_r_x = (outer_derivative_2 * inner_derivative_1_x.col(0) * inner_derivative_1_r).matrix().sum();
      double derivative_r_y = (outer_derivative_2 * inner_derivative_1_x.col(1) * inner_derivative_1_r).matrix().sum();
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
            ((xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) / bandwidth;
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
              ((xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) / bandwidth;
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
              ((xy.matrix().rowwise() - next_center).rowwise().norm().array() - next_radius) / bandwidth;
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

      if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(radius) || center[0] < break_min_x ||
          center[0] > break_max_x || center[1] < break_min_y || center[1] > break_max_y || radius < break_min_radius ||
          radius > break_max_radius || radius <= 0 || iteration == max_iterations - 1) {
        diverged = true;
        break;
      }

      if ((abs(radius_update) < break_min_change) && (abs(center_update[0]) < break_min_change) &&
          (abs(center_update[1]) < break_min_change)) {
        break;
      }
    }

    if (!diverged && fitting_score >= min_fitting_score && std::isfinite(center[0]) && std::isfinite(center[1]) &&
        std::isfinite(radius) && radius > 0) {
      fitted_circles(idx, 0) = center[0];
      fitted_circles(idx, 1) = center[1];
      fitted_circles(idx, 2) = radius;
      fitting_converged(idx) = true;
      fitting_losses(idx) = -1 * fitting_score;
    }
  }

  std::vector<int> converged_indices{};
  for (int i = 0; i < fitting_converged.size(); ++i) {
    if (fitting_converged[i]) {
      converged_indices.push_back(i);
    }
  }

  return std::make_tuple(fitted_circles(converged_indices, Eigen::all), fitting_losses(converged_indices));
}
}  // namespace CircleDetection
