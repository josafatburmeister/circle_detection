#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CircleDetection/circle_detection.h"

PYBIND11_MODULE(_circle_detection_cpp, m) {
  m.doc() = R"pbdoc(
    Circle detection in 2D point sets.
  )pbdoc";

  m.def("detect_circles", &CircleDetection::detect_circles, pybind11::return_value_policy::reference_internal,
        R"pbdoc(
    C++ implementation of the M-estimator-based circle detection method proposed by Tim Garlipp and Christine H.
    Müller. For more details, see the documentation of the Python wrapper method
    :code:`circle_detection.detect_circles()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
