#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CircleDetection/operations.h"

PYBIND11_MODULE(
    _operations_cpp,
    m)  // NOLINT(readability-named-parameter,hicpp-named-parameter,misc-use-internal-linkage)
{
  m.doc() = R"pbdoc(
    Post-processing operations for the circle detection.
  )pbdoc";

  m.def(
      "non_maximum_suppression", &CircleDetection::non_maximum_suppression<float>,
      pybind11::return_value_policy::take_ownership, "");

  m.def(
      "non_maximum_suppression", &CircleDetection::non_maximum_suppression<double>,
      pybind11::return_value_policy::take_ownership,
      R"pbdoc(
    Non-maximum suppression for overlapping circles. For more details, see the documentation of the Python wrapper
    method :code:`circle_detection.operations.non_maximum_suppression()`.
  )pbdoc");

  m.def(
      "circumferential_completeness_index", &CircleDetection::circumferential_completeness_index<float>,
      pybind11::return_value_policy::take_ownership, "");

  m.def(
      "circumferential_completeness_index", &CircleDetection::circumferential_completeness_index<double>,
      pybind11::return_value_policy::take_ownership,
      R"pbdoc(
    Calculates the circumferential completeness indices of the specified circles. For more details, see the documentation of the Python wrapper
    method :code:`circle_detection.operations.circumferential_completeness_index()`.
  )pbdoc");

  m.def(
      "filter_circumferential_completeness_index", &CircleDetection::filter_circumferential_completeness_index<float>,
      pybind11::return_value_policy::take_ownership, "");

  m.def(
      "filter_circumferential_completeness_index", &CircleDetection::filter_circumferential_completeness_index<double>,
      pybind11::return_value_policy::take_ownership,
      R"pbdoc(
    Filters out the circles whose circumferential completeness index is below the specified minimum circumferential
    completeness index. For more details, see the documentation of the Python wrapper
    method :code:`circle_detection.operations.filter_circumferential_completeness_index()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
