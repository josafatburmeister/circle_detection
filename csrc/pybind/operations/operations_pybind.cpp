#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "CircleDetection/operations/non_maximum_suppression.h"

PYBIND11_MODULE(_operations_cpp,
                m)  // NOLINT(readability-named-parameter,hicpp-named-parameter,misc-use-internal-linkage)
{
  m.doc() = R"pbdoc(
    Post-processing operations for the circle detection.
  )pbdoc";

  m.def("non_maximum_suppression", &CircleDetection::non_maximum_suppression,
        pybind11::return_value_policy::reference_internal,
        R"pbdoc(
    Non-maximum suppression for overlapping circles. For more details, see the documentation of the Python wrapper
    method :code:`circle_detection.operations.non_maximum_suppression()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
