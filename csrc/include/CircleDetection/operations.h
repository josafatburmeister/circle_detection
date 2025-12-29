#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "type_aliases.h"

#ifndef OPERATIONS_H
#define OPERATIONS_H

namespace CircleDetection {

template <typename scalar_T>
scalar_T stddev(RefArrayX2<scalar_T> x) {
  scalar_T variance = (x - x.mean()).square().mean();
  return std::sqrt(variance);
}

template <typename scalar_T>
ArrayX<scalar_T> circumferential_completeness_index(
    RefArrayX3<scalar_T> circles,
    RefArrayX2<scalar_T> xy,
    RefArrayXl batch_lengths_circles,
    RefArrayXl batch_lengths_xy,
    int64_t num_regions,
    scalar_T max_dist,
    int num_workers = 1) {
  std::cout << "step 1" << std::endl;
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

  Eigen::setNbThreads(1);
  Eigen::initParallel();

  constexpr scalar_T PI = 3.14159265358979311600;

  std::cout << "step 2" << std::endl;

  int64_t num_batches = batch_lengths_circles.size();
  ArrayX<scalar_T> circumferential_completeness_indices(circles.rows());

  scalar_T angular_step_size = 2 * PI / static_cast<scalar_T>(num_regions);

  ArrayXl batch_starts_circles(num_batches);
  ArrayXl batch_starts_xy(num_batches);
  ArrayXl batch_indices(circles.rows());

  int64_t batch_start_circles = 0;
  int64_t batch_start_xy = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts_circles(batch_idx) = batch_start_circles;
    batch_starts_xy(batch_idx) = batch_start_xy;
    batch_indices(Eigen::seqN(batch_start_circles, batch_lengths_circles(batch_idx))) = batch_idx;
    batch_start_circles += batch_lengths_circles(batch_idx);
    batch_start_xy += batch_lengths_xy(batch_idx);
  }
  std::cout << "step 3" << std::endl;
  std::cout << "batch_lengths_xy " << batch_lengths_xy << std::endl;
  std::cout << "batch_starts_xy " << batch_starts_xy << std::endl;
  std::cout << "batch_indices" << batch_indices << std::endl;

  std::cout << "num_workers" << num_workers << std::endl;

  #pragma omp parallel for default(shared) num_threads(num_workers)
  for (int64_t idx = 0; idx < circles.rows(); ++idx) {
    std::cout << "step 4 " << idx << std::endl;
    int64_t batch_idx = batch_indices(idx);
    std::cout << "batch_lengths_xy " << batch_lengths_xy << std::endl;
    std::cout << "batch_lengths_xy(batch_idx) " << batch_lengths_xy(batch_idx) << std::endl;
    std::cout << "batch_idx" << batch_idx << std::endl;
    Eigen::RowVector3<scalar_T> circle = circles(idx, Eigen::all);
    std::cout << "step 5 " << idx << std::endl;
    std::cout << "circle " << circle << std::endl;
    std::cout << "batch_idx " << batch_idx << std::endl;
    std::cout << "batch_starts_xy(batch_idx) " << batch_starts_xy(batch_idx) << std::endl;
    std::cout << "test" << std::endl;
    std::cout << "batch_idx " << batch_idx << std::endl;

    auto indexer = Eigen::seqN(batch_starts_xy(batch_idx), batch_lengths_xy(batch_idx));

    std::cout << "indexer " << std::endl;

    auto current_xy = xy(indexer, Eigen::all);

    std::cout << "step 6.1 " << idx << std::endl;
    std::cout << "circle({0, 1} " << circle({0, 1}) << std::endl;

    ArrayX2<scalar_T> centered_xy = current_xy.rowwise() - circle({0, 1}).array();
    std::cout << "step 6.2 " << idx << std::endl;
    ArrayX<scalar_T> radii = centered_xy.rowwise().norm();
    std::cout << "step 6.3 " << idx << std::endl;

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
      ArrayX2<scalar_T> circle_xy = centered_xy(circle_xy_indices, Eigen::all);

      std::cout << "step 7 " << idx << std::endl;

      ArrayX<scalar_T> angles =
          circle_xy(Eigen::all, 1).binaryExpr(circle_xy(Eigen::all, 0), [](scalar_T y, scalar_T x) {
            return std::atan2(y, x);
          });

      ArrayXl sections =
          (angles / angular_step_size).floor().unaryExpr([](scalar_T x) { return static_cast<int64_t>(x); });
      sections = sections.unaryExpr([num_regions](const int64_t x) { return x % num_regions; });

      std::set<int64_t> filled_sections(sections.data(), sections.data() + sections.size());

      circumferential_completeness_indices(idx) = filled_sections.size() / static_cast<scalar_T>(num_regions);
    }
  }
  std::cout << "step 4" << std::endl;

  return circumferential_completeness_indices;
}

template <typename scalar_T>
std::tuple<ArrayX3<scalar_T>, ArrayXl, ArrayXl> filter_circumferential_completeness_index(
    RefArrayX3<scalar_T> circles,
    RefArrayX2<scalar_T> xy,
    RefArrayXl batch_lengths_circles,
    RefArrayXl batch_lengths_xy,
    int64_t num_regions,
    scalar_T max_dist,
    scalar_T min_circumferential_completeness_index,
    int num_workers = 1) {
  ArrayX<scalar_T> circumferential_completeness_indices = circumferential_completeness_index<scalar_T>(
      circles, xy, batch_lengths_circles, batch_lengths_xy, num_regions, max_dist, num_workers);
  int64_t num_batches = batch_lengths_circles.size();

  std::vector<int64_t> filtered_indices = {};
  ArrayXl filtered_batch_lengths_circles = ArrayXl::Constant(num_batches, 0);

  ArrayXl batch_indices(circles.rows());

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

  ArrayXl filtered_indices_array = Eigen::Map<ArrayXl>(filtered_indices.data(), filtered_indices.size());

  return std::make_tuple(circles(filtered_indices, Eigen::all), filtered_batch_lengths_circles, filtered_indices_array);
}

template <typename scalar_T>
std::tuple<ArrayX3<scalar_T>, ArrayX<scalar_T>, ArrayXl, ArrayXl> non_maximum_suppression(
    RefArrayX3<scalar_T> circles, RefArrayX<scalar_T> fitting_scores, RefArrayXl batch_lengths, int num_workers = 1) {
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

  ArrayXl batch_starts(num_batches);

  int64_t batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts(batch_idx) = batch_start;
    batch_start += batch_lengths(batch_idx);
  }

  std::vector<std::vector<int64_t>> selected_indices(num_batches);
  ArrayXl new_batch_lengths = ArrayXl::Constant(num_batches, 0);
  ArrayXl new_batch_starts = ArrayXl::Constant(num_batches, 0);

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
      Vector2<scalar_T> center(circles(batch_start + current_idx, 0), circles(batch_start + current_idx, 1));
      auto radius = circles(batch_start + current_idx, 2);

      auto iter = sorted_indices.begin();
      while (iter < sorted_indices.end()) {
        auto other_idx = *iter;
        Vector2<scalar_T> other_center(circles(batch_start + other_idx, 0), circles(batch_start + other_idx, 1));
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

  ArrayX3<scalar_T> selected_circles(total_num_selected_circles, 3);
  ArrayX<scalar_T> selected_fitting_scores(total_num_selected_circles);
  ArrayXl selected_indices_array(total_num_selected_circles);

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
        Eigen::Map<ArrayXl>(selected_indices[batch_idx].data(), new_batch_lengths(batch_idx));
  }

  return std::make_tuple(selected_circles, selected_fitting_scores, new_batch_lengths, selected_indices_array);
}

}  // namespace CircleDetection

#endif  // OPERATIONS_H
