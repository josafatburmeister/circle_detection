#include <Eigen/Dense>
#include <cstdint>

#ifndef TYPE_ALIASES_H
#define TYPE_ALIASES_H

namespace CircleDetection {

// type aliases for better code readability

using ArrayXl = Eigen::Array<int64_t, Eigen::Dynamic, 1>;
using RefArrayXl = Eigen::Ref<ArrayXl>;

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using RefArrayXb = Eigen::Ref<ArrayXb>;

template <typename scalar_T>
using ArrayX = Eigen::Array<scalar_T, Eigen::Dynamic, 1>;

template <typename scalar_T>
using RefArrayX = Eigen::Ref<ArrayX<scalar_T>>;

template <typename scalar_T>
using ArrayX2 = Eigen::Array<scalar_T, Eigen::Dynamic, 2>;

template <typename scalar_T>
using RefArrayX2 = Eigen::Ref<ArrayX2<scalar_T>>;

template <typename scalar_T>
using ArrayX3 = Eigen::Array<scalar_T, Eigen::Dynamic, 3>;

template <typename scalar_T>
using RefArrayX3 = Eigen::Ref<ArrayX3<scalar_T>>;

template <typename scalar_T>
using ArrayX5 = Eigen::Array<scalar_T, Eigen::Dynamic, 5>;

template <typename scalar_T>
using RefArrayX5 = Eigen::Ref<ArrayX5<scalar_T>>;

template <typename scalar_T>
using MatrixX2 = Eigen::Matrix<scalar_T, Eigen::Dynamic, 2>;

template <typename scalar_T>
using MatrixX3 = Eigen::Matrix<scalar_T, Eigen::Dynamic, 3>;

template <typename scalar_T>
using RowVector2 = Eigen::RowVector<scalar_T, 2>;

template <typename scalar_T>
using Vector2 = Eigen::Vector<scalar_T, 2>;

template <typename scalar_T>
using Vector3 = Eigen::Vector<scalar_T, 3>;
}  // namespace CircleDetection

#endif  // TYPE_ALIASES_H
