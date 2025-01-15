#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CircleDetection/circle_detection_m_estimator.h"
#include "CircleDetection/circle_detection_ransac.h"

PYBIND11_MODULE(_circle_detection_cpp, m) {
  m.doc() = R"pbdoc(
    Circle detection in 2D point sets.
  )pbdoc";

  m.def(
      "detect_circles_m_estimator", &CircleDetection::detect_circles_m_estimator<float>,
      pybind11::return_value_policy::reference_internal, "");

  m.def(
      "detect_circles_m_estimator", &CircleDetection::detect_circles_m_estimator<double>,
      pybind11::return_value_policy::reference_internal,
      R"pbdoc(
    C++ implementation of the M-estimator-based circle detection method proposed by Tim Garlipp and Christine H.
    Müller. For more details, see the documentation of the Python wrapper class
    :code:`circle_detection.MEstimator`.
  )pbdoc");

  m.def(
      "detect_circles_ransac", &CircleDetection::detect_circles_ransac<float>,
      pybind11::return_value_policy::reference_internal, "");

  m.def(
      "detect_circles_ransac", &CircleDetection::detect_circles_ransac<double>,
      pybind11::return_value_policy::reference_internal,
      R"pbdoc(
    C++ implementation of RANSAC circle detection that is based on least-squares circle fitting. For more details, see
    the documentation of the Python wrapper class :code:`circle_detection.Ransac`.
  )pbdoc");

  m.def(
      "fit_circle_lsq", &CircleDetection::fit_circle_lsq<float>, pybind11::return_value_policy::reference_internal, "");

  m.def(
      "fit_circle_lsq", &CircleDetection::fit_circle_lsq<double>, pybind11::return_value_policy::reference_internal,
      R"pbdoc(
    C++ implementation of least-squares circle fitting. For more details, see the documentation of the Python wrapper
    method :code:`circle_detection.fit_circle_lsq()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
