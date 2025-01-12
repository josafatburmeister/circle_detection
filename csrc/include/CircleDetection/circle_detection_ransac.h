#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "loss_functions.h"

#ifndef CIRCLE_DETECTION_RANSAC_H
#define CIRCLE_DETECTION_RANSAC_H

namespace {

template <typename scalar_T>
scalar_T stddev(Eigen::Array<scalar_T, Eigen::Dynamic, 2> x) {
  scalar_T variance = (x - x.mean()).square().mean();
  return std::sqrt(variance);
}
}  // namespace

namespace CircleDetection {

template <typename scalar_T>
Eigen::Vector<scalar_T, 3> fit_circle_lsq(Eigen::Array<scalar_T, Eigen::Dynamic, 2> xy) {
  Eigen::Vector<scalar_T, 2> origin = xy.colwise().mean().matrix();
  xy = xy.rowwise() - origin.transpose().array();

  scalar_T scale = stddev(xy);

  xy = xy / scale;

  Eigen::Matrix<scalar_T, Eigen::Dynamic, 3> A(xy.rows(), 3);
  A(Eigen::all, {0, 1}) = xy * 2;
  A(Eigen::all, {2}) = Eigen::Array<scalar_T, Eigen::Dynamic, 1>::Constant(xy.rows(), 1.0);
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> f = xy.rowwise().squaredNorm();

  auto qr = A.fullPivHouseholderQr();

  if (qr.rank() != 3) {
    return Eigen::Vector<scalar_T, 3>::Constant(3, -1);
  }

  Eigen::Array<scalar_T, Eigen::Dynamic, 1> lsq_solution = qr.solve(f.matrix()).array();

  Eigen::Vector<scalar_T, 2> center = lsq_solution({0, 1});
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> squared_dists =
      (xy.rowwise() - center.transpose().array()).rowwise().squaredNorm();
  scalar_T radius = std::sqrt(squared_dists.mean());

  Eigen::Vector<scalar_T, 3> circle;

  circle({0, 1}) = center;
  circle(2) = radius;
  circle *= scale;
  circle({0, 1}) += origin;

  return circle;
}

template <typename scalar_T>
std::tuple<Eigen::Array<scalar_T, Eigen::Dynamic, 3>, Eigen::Array<scalar_T, Eigen::Dynamic, 1>,
           Eigen::Array<int64_t, Eigen::Dynamic, 1>>
detect_circles_ransac(
    Eigen::Array<scalar_T, Eigen::Dynamic, 2> xy, Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_lengths,
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_min_x, Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_max_x,
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_min_y, Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_max_y,
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_min_radius,
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> break_max_radius, scalar_T bandwidth, int iterations, int num_samples,
    int min_concensus_points = 3, scalar_T min_fitting_score = 1e-6, int num_workers = 1, int seed = -1) {
  if (num_samples < 3) {
    throw std::invalid_argument("The required number of hypothetical inlier points must be at least 3.");
  }

  if (min_concensus_points < 3) {
    throw std::invalid_argument("The required number of consensus points must be at least 3.");
  }

  if (xy.rows() != batch_lengths.sum()) {
    throw std::invalid_argument("The number of points must be equal to the sum of batch_lengths");
  }

  if (break_min_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_min_x must be equal to the batch size.");
  }
  if (break_max_x.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_max_x must be equal to the batch size.");
  }
  if (break_min_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_min_y must be equal to the batch size.");
  }
  if (break_max_y.rows() != batch_lengths.rows()) {
    throw std::invalid_argument("The length of break_max_y must be equal to the batch size.");
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

  int64_t num_batches = batch_lengths.rows();
  int64_t num_circles = num_batches * iterations;
  Eigen::Array<scalar_T, Eigen::Dynamic, 3> circles =
      Eigen::Array<scalar_T, Eigen::Dynamic, 3>::Constant(num_circles, 3, -1);
  Eigen::Array<bool, Eigen::Dynamic, 1> diverged = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_circles, true);
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> fitting_scores =
      Eigen::Array<scalar_T, Eigen::Dynamic, 1>::Constant(num_circles, -1);
  std::vector<std::mt19937> random_generators(num_batches);

  if (seed == -1) {
    std::random_device random_device;
    seed = random_device();
  }

  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_starts(num_batches);

  int64_t batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts(batch_idx) = batch_start;
    batch_start += batch_lengths(batch_idx);
    random_generators.push_back(std::mt19937(seed));
  }

#pragma omp parallel num_threads(num_workers)
#pragma omp single
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    if (batch_lengths(batch_idx) < 3) {
      continue;
    }
    for (int64_t i = 0; i < iterations; ++i) {
#pragma omp task
      {
        Eigen::Array<scalar_T, Eigen::Dynamic, 2> current_xy =
            xy(seqN(batch_starts(batch_idx), batch_lengths(batch_idx)), Eigen::all);
        int samples_to_draw = std::min(num_samples, static_cast<int>(current_xy.rows()));

        std::vector<int64_t> indices(current_xy.rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(), random_generators[batch_idx]);

        std::vector<int64_t> hypothetical_inliers_indices(indices.begin(), indices.begin() + samples_to_draw);

        Eigen::Array<scalar_T, Eigen::Dynamic, 2> hypothetical_inliers_xy =
            current_xy(hypothetical_inliers_indices, Eigen::all);

        Eigen::Vector<scalar_T, 3> circle = fit_circle_lsq<scalar_T>(hypothetical_inliers_xy);

        for (int step = 0; step < 1;
             ++step) {  // we use a for loop with a single iteration so that we can use break to exit early
          if (circle(2) == -1 || circle(0) < break_min_x(batch_idx) || circle(0) > break_max_x(batch_idx) ||
              circle(1) < break_min_y(batch_idx) || circle(1) > break_max_y(batch_idx) ||
              circle(2) < break_min_radius(batch_idx) || circle(2) > break_max_radius(batch_idx)) {
            break;
          }
          // dists to circle center
          Eigen::Array<scalar_T, Eigen::Dynamic, 1> dists_to_circle =
              (current_xy.rowwise() - circle({0, 1}).transpose().array()).rowwise().norm();
          // dists to circle outline
          dists_to_circle = (dists_to_circle - circle(2)).abs();

          std::vector<int64_t> consensus_indices;
          for (int64_t j = 0; j < current_xy.rows(); ++j) {
            if (dists_to_circle(j) <= bandwidth) {
              consensus_indices.push_back(j);
            }
          }
          if (consensus_indices.size() < min_concensus_points) {
            break;
          }
          Eigen::Array<scalar_T, Eigen::Dynamic, 2> consensus_xy = current_xy(consensus_indices, Eigen::all);

          // fit circle to all consensus points
          circle = fit_circle_lsq<scalar_T>(consensus_xy);

          if (circle(2) == -1) {
            break;
          }

          // dists to circle center
          dists_to_circle = (current_xy.rowwise() - circle({0, 1}).transpose().array()).rowwise().norm();
          // dists to circle outline
          dists_to_circle = (dists_to_circle - circle(2)).abs();

          scalar_T fitting_score =
              1 / bandwidth *
              (dists_to_circle / bandwidth).unaryExpr(&CircleDetection::score_fn_scalar<scalar_T>).sum();

          if (fitting_score < min_fitting_score) {
            break;
          }

          int64_t flat_idx = batch_idx * iterations + i;

          diverged(flat_idx) = false;
          circles(flat_idx, Eigen::all) = circle;
          fitting_scores(flat_idx) = fitting_score;
        }
      }
    }
  }

  std::vector<int64_t> selected_indices;
  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_lengths_circles =
      Eigen::Array<int64_t, Eigen::Dynamic, 1>::Constant(num_batches, 0);

  for (int64_t i = 0; i < num_batches; ++i) {
    for (int64_t j = 0; j < iterations; ++j) {
      int64_t flat_idx = i * iterations + j;
      if (!diverged(flat_idx)) {
        selected_indices.push_back(flat_idx);
        batch_lengths_circles(i) += 1;
      }
    }
  }

  return std::make_tuple(circles(selected_indices, Eigen::all), fitting_scores(selected_indices),
                         batch_lengths_circles);
}
}  // namespace CircleDetection

#endif  // CIRCLE_DETECTION_RANSAC_H
