
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Eigen;

namespace CircleDetection {
std::tuple<ArrayX3d, ArrayXd> non_maximum_suppression(ArrayX3d circles, ArrayXd fitting_scores) {
  std::vector<int> kept_indices = {};
  std::vector<int> sorted_indices(circles.rows());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&fitting_scores](int i, int j) { return fitting_scores(i) < fitting_scores(j); });

  while (sorted_indices.size() > 0) {
    auto current_idx = sorted_indices[0];
    sorted_indices.erase(sorted_indices.begin());
    kept_indices.push_back(current_idx);
    Vector2d center(circles(current_idx, 0), circles(current_idx, 1));
    auto radius = circles(current_idx, 2);

    auto iter = sorted_indices.begin();
    while (iter < sorted_indices.end()) {
      auto other_idx = *iter;
      Vector2d other_center(circles(other_idx, 0), circles(other_idx, 1));
      auto other_radius = circles(other_idx, 2);

      if ((center - other_center).norm() < radius + other_radius) {
        iter = sorted_indices.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  return std::make_tuple(circles(kept_indices, Eigen::all), fitting_scores(kept_indices));
}
}  // namespace CircleDetection
