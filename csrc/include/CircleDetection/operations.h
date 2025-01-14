#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#ifndef OPERATIONS_H
#define OPERATIONS_H

namespace CircleDetection {

template <typename scalar_T>
scalar_T stddev(Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 2>> x) {
  scalar_T variance = (x - x.mean()).square().mean();
  return std::sqrt(variance);
}

template <typename scalar_T>
Eigen::Array<scalar_T, Eigen::Dynamic, 1> circumferential_completeness_index(
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 3>> circles,
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 2>> xy,
    Eigen::Ref<Eigen::Array<int64_t, Eigen::Dynamic, 1>> batch_lengths_circles,
    Eigen::Ref<Eigen::Array<int64_t, Eigen::Dynamic, 1>> batch_lengths_xy,
    int64_t num_regions,
    scalar_T max_dist,
    int num_workers = 1) {
  if (batch_lengths_circles.rows() != batch_lengths_xy.rows()) {
    throw std::invalid_argument("The length of batch_lengths_circles and batch_lengths_xy must be equal.");
  }
  if (circles.rows() != batch_lengths_circles.sum()) {
    throw std::invalid_argument("The number of circles must be equal to the sum of batch_lengths_circles.");
  }
  if (xy.rows() != batch_lengths_xy.sum()) {
    throw std::invalid_argument("The number of points must be equal to the sum of batch_lengths_xy.");
  }

  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  constexpr scalar_T PI = 3.14159265358979311600;

  int64_t num_batches = batch_lengths_circles.size();
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> circumferential_completeness_indices(circles.rows());

  scalar_T angular_step_size = 2 * PI / static_cast<scalar_T>(num_regions);

  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_starts_circles(num_batches);
  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_starts_xy(num_batches);
  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_indices(circles.rows());

  int64_t batch_start_circles = 0;
  int64_t batch_start_xy = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts_circles(batch_idx) = batch_start_circles;
    batch_starts_xy(batch_idx) = batch_start_xy;
    batch_indices(Eigen::seqN(batch_start_circles, batch_lengths_circles(batch_idx))) = batch_idx;
    batch_start_circles += batch_lengths_circles(batch_idx);
    batch_start_xy += batch_lengths_xy(batch_idx);
  }

#pragma omp parallel for default(shared) num_threads(num_workers)
  for (int64_t idx = 0; idx < circles.rows(); ++idx) {
    int64_t batch_idx = batch_indices(idx);
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> circle = circles(idx, Eigen::all);

    Eigen::Array<scalar_T, Eigen::Dynamic, 2> centered_xy =
        xy(Eigen::seqN(batch_starts_xy(batch_idx), batch_lengths_xy(batch_idx)), Eigen::all).rowwise() -
        circle({0, 1}).transpose();
    Eigen::Array<scalar_T, Eigen::Dynamic, 1> radii = centered_xy.rowwise().norm();

    if (centered_xy.rows() == 0) {
      circumferential_completeness_indices(idx) = 0.0;
    } else {
      std::vector<int64_t> circle_xy_indices;
      if (max_dist < 0) {
        for (int64_t i = 0; i < radii.rows(); ++i) {
          if (radii(i) >= 0.7 * circle(2) && radii(i) <= 1.3 * circle(2)) {
            circle_xy_indices.push_back(i);
          }
        }
      } else {
        for (int64_t i = 0; i < radii.rows(); ++i) {
          if (std::abs(radii(i) - circle(2)) <= max_dist) {
            circle_xy_indices.push_back(i);
          }
        }
      }
      Eigen::Array<scalar_T, Eigen::Dynamic, 2> circle_xy = centered_xy(circle_xy_indices, Eigen::all);

      Eigen::Array<scalar_T, Eigen::Dynamic, 1> angles =
          circle_xy(Eigen::all, 1).binaryExpr(circle_xy(Eigen::all, 0), [](scalar_T y, scalar_T x) {
            return std::atan2(y, x);
          });

      Eigen::Array<int64_t, Eigen::Dynamic, 1> sections =
          (angles / angular_step_size).floor().unaryExpr([](scalar_T x) { return static_cast<int64_t>(x); });
      sections = sections.unaryExpr([num_regions](const int64_t x) { return x % num_regions; });

      std::set<int64_t> filled_sections(sections.data(), sections.data() + sections.size());

      circumferential_completeness_indices(idx) = filled_sections.size() / static_cast<scalar_T>(num_regions);
    }
  }

  return circumferential_completeness_indices;
}

template <typename scalar_T>
std::tuple<
    Eigen::Array<scalar_T, Eigen::Dynamic, 3>,
    Eigen::Array<int64_t, Eigen::Dynamic, 1>,
    Eigen::Array<int64_t, Eigen::Dynamic, 1>>
filter_circumferential_completeness_index(
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 3>> circles,
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 2>> xy,
    Eigen::Ref<Eigen::Array<int64_t, Eigen::Dynamic, 1>> batch_lengths_circles,
    Eigen::Ref<Eigen::Array<int64_t, Eigen::Dynamic, 1>> batch_lengths_xy,
    int64_t num_regions,
    scalar_T max_dist,
    scalar_T min_circumferential_completeness_index,
    int num_workers = 1) {
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> circumferential_completeness_indices =
      circumferential_completeness_index<scalar_T>(
          circles, xy, batch_lengths_circles, batch_lengths_xy, num_regions, max_dist, num_workers);
  int64_t num_batches = batch_lengths_circles.size();

  std::vector<int64_t> filtered_indices = {};
  Eigen::Array<int64_t, Eigen::Dynamic, 1> filtered_batch_lengths_circles =
      Eigen::Array<int64_t, Eigen::Dynamic, 1>::Constant(num_batches, 0);

  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_indices(circles.rows());

  int64_t batch_start_circles = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_indices(Eigen::seqN(batch_start_circles, batch_lengths_circles(batch_idx))) = batch_idx;
    batch_start_circles += batch_lengths_circles(batch_idx);
  }

  for (int64_t i = 0; i < circles.rows(); ++i) {
    if (circumferential_completeness_indices(i) >= min_circumferential_completeness_index) {
      filtered_indices.push_back(i);
      filtered_batch_lengths_circles(batch_indices(i)) += 1;
    }
  }

  Eigen::Array<int64_t, Eigen::Dynamic, 1> filtered_indices_array =
      Eigen::Map<Eigen::Array<int64_t, Eigen::Dynamic, 1>>(filtered_indices.data(), filtered_indices.size());

  return std::make_tuple(circles(filtered_indices, Eigen::all), filtered_batch_lengths_circles, filtered_indices_array);
}

template <typename scalar_T>
std::tuple<
    Eigen::Array<scalar_T, Eigen::Dynamic, 3>,
    Eigen::Array<scalar_T, Eigen::Dynamic, 1>,
    Eigen::Array<int64_t, Eigen::Dynamic, 1>,
    Eigen::Array<int64_t, Eigen::Dynamic, 1>>
non_maximum_suppression(
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 3>> circles,
    Eigen::Ref<Eigen::Array<scalar_T, Eigen::Dynamic, 1>> fitting_scores,
    Eigen::Ref<Eigen::Array<int64_t, Eigen::Dynamic, 1>> batch_lengths,
    int num_workers = 1) {
  if (circles.rows() != fitting_scores.rows()) {
    throw std::invalid_argument("circles and fitting_scores must have the same number of entries.");
  }

  if (circles.rows() != batch_lengths.sum()) {
    throw std::invalid_argument("The number of circles must be equal to the sum of batch_lengths.");
  }

  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  int64_t num_batches = batch_lengths.rows();

  Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_starts(num_batches);

  int64_t batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts(batch_idx) = batch_start;
    batch_start += batch_lengths(batch_idx);
  }

  std::vector<std::vector<int64_t>> selected_indices(num_batches);
  Eigen::Array<int64_t, Eigen::Dynamic, 1> new_batch_lengths =
      Eigen::Array<int64_t, Eigen::Dynamic, 1>::Constant(num_batches, 0);
  Eigen::Array<int64_t, Eigen::Dynamic, 1> new_batch_starts =
      Eigen::Array<int64_t, Eigen::Dynamic, 1>::Constant(num_batches, 0);

#pragma omp parallel for default(shared) num_threads(num_workers)
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    int64_t batch_start = batch_starts(batch_idx);
    int64_t num_circles = circles(Eigen::seqN(batch_start, batch_lengths(batch_idx)), Eigen::all).rows();
    std::vector<int64_t> sorted_indices(num_circles);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&fitting_scores, batch_start](int64_t i, int64_t j) {
      return fitting_scores(batch_start + i) > fitting_scores(batch_start + j);
    });

    while (sorted_indices.size() > 0) {
      auto current_idx = sorted_indices[0];
      sorted_indices.erase(sorted_indices.begin());
      selected_indices[batch_idx].push_back(batch_start + current_idx);
      new_batch_lengths(batch_idx) += 1;
      Eigen::Vector2d center(circles(batch_start + current_idx, 0), circles(batch_start + current_idx, 1));
      auto radius = circles(batch_start + current_idx, 2);

      auto iter = sorted_indices.begin();
      while (iter < sorted_indices.end()) {
        auto other_idx = *iter;
        Eigen::Vector2d other_center(circles(batch_start + other_idx, 0), circles(batch_start + other_idx, 1));
        auto other_radius = circles(batch_start + other_idx, 2);

        if ((center - other_center).norm() < radius + other_radius) {
          iter = sorted_indices.erase(iter);
        } else {
          ++iter;
        }
      }
    }
  }

  int64_t total_num_selected_circles = new_batch_lengths.sum();

  Eigen::Array<scalar_T, Eigen::Dynamic, 3> selected_circles(total_num_selected_circles, 3);
  Eigen::Array<scalar_T, Eigen::Dynamic, 1> selected_fitting_scores(total_num_selected_circles);
  Eigen::Array<int64_t, Eigen::Dynamic, 1> selected_indices_array(total_num_selected_circles);

  batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    new_batch_starts(batch_idx) = batch_start;
    batch_start += new_batch_lengths(batch_idx);
  }

#pragma omp parallel for default(shared) num_threads(num_workers)
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    selected_circles(Eigen::seqN(new_batch_starts(batch_idx), new_batch_lengths(batch_idx)), Eigen::all) =
        circles(selected_indices[batch_idx], Eigen::all);
    selected_fitting_scores(Eigen::seqN(new_batch_starts(batch_idx), new_batch_lengths(batch_idx))) =
        fitting_scores(selected_indices[batch_idx]);
    selected_indices_array(Eigen::seqN(new_batch_starts(batch_idx), new_batch_lengths(batch_idx))) =
        Eigen::Map<Eigen::Array<int64_t, Eigen::Dynamic, 1>>(
            selected_indices[batch_idx].data(), new_batch_lengths(batch_idx));
  }

  return std::make_tuple(selected_circles, selected_fitting_scores, new_batch_lengths, selected_indices_array);
}

}  // namespace CircleDetection

#endif  // OPERATIONS_H
