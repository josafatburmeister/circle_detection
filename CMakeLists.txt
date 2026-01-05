cmake_minimum_required(VERSION 3.30...3.35)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

option(BUILD_PYTHON_BINDINGS "Build the Python bindings." ON)
option(BUILD_LIB "Build the header-only library." ON)

# Read the package version from pyproject.toml using a regular expression
set(PYPROJECT_FILE "${CMAKE_SOURCE_DIR}/pyproject.toml")
file(READ "${PYPROJECT_FILE}" PYPROJECT_CONTENTS)

string(REGEX MATCH "version[ \t]*=[ \t]*\"([0-9]+\.[0-9]+\.[0-9]+)\"" VERSION_MATCH "${PYPROJECT_CONTENTS}")
if(VERSION_MATCH)
    string(REGEX REPLACE "version[ \t]*=[ \t]*\"([0-9]+\.[0-9]+\.[0-9]+)\"" "\\1" PROJECT_VERSION "${VERSION_MATCH}")
else()
    message(FATAL_ERROR "Could not extract package version from pyproject.toml")
endif()

if(APPLE)

  set(OPENMP_ROOT "" CACHE PATH "Prefix where libomp is installed (has include/ and lib/)")

  if(OPENMP_ROOT)
    message(STATUS "Using OPENMP_ROOT=${OPENMP_ROOT}")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${OPENMP_ROOT}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${OPENMP_ROOT}/lib/libomp.dylib")
  else()
    execute_process(COMMAND brew --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Using Homebrew OpenMP from ${BREW_PREFIX}/opt/libomp")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${BREW_PREFIX}/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${BREW_PREFIX}/opt/libomp/lib/libomp.dylib")
  endif()
endif()

find_package(OpenMP REQUIRED)

set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip")
set(EIGEN_SOURCE_DIR "${CMAKE_SOURCE_DIR}/eigen3")
set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen3/eigen-3.4.0")
set(EIGEN_ZIP_PATH "${EIGEN_SOURCE_DIR}/eigen-3.4.0.zip")
set(EIGEN_CORE_HEADER_PATH "${EIGEN_INCLUDE_DIR}/Eigen/Core")

if (NOT EXISTS ${EIGEN_CORE_HEADER_PATH})
  message(STATUS "Downloading Eigen.")
  file(MAKE_DIRECTORY ${EIGEN_SOURCE_DIR})
  file(DOWNLOAD ${EIGEN_URL} ${EIGEN_ZIP_PATH} SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${EIGEN_ZIP_PATH} DESTINATION ${EIGEN_SOURCE_DIR})
else()
    message(STATUS "Found Eigen source directory. Skipping download.")
endif()

include_directories( ${EIGEN_INCLUDE_DIR} )

if(BUILD_LIB)
  set(LIB_NAME CircleDetection)

  add_library(${LIB_NAME} INTERFACE)

  target_include_directories(${LIB_NAME} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/csrc/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  target_compile_features(${LIB_NAME} INTERFACE cxx_std_17)

  install(TARGETS ${LIB_NAME} EXPORT ${LIB_NAME}Targets)

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${LIB_NAME}ConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${LIB_NAME}Config.cmake.in
    "${PROJECT_BINARY_DIR}/${LIB_NAME}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${LIB_NAME}/cmake
  )

  install(FILES "${PROJECT_BINARY_DIR}/${LIB_NAME}Config.cmake"
                "${PROJECT_BINARY_DIR}/${LIB_NAME}ConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${LIB_NAME}/cmake)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/csrc/include DESTINATION .)

endif()

if(BUILD_PYTHON_BINDINGS)

  include_directories( ${PROJECT_SOURCE_DIR}/csrc/include )

  set(PYBIND11_NEWPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_circle_detection_cpp MODULE csrc/pybind/circle_detection_pybind.cpp)
  target_link_libraries(_circle_detection_cpp PRIVATE OpenMP::OpenMP_CXX)

  pybind11_add_module(_operations_cpp MODULE csrc/pybind/operations_pybind.cpp)
  target_link_libraries(_operations_cpp PRIVATE OpenMP::OpenMP_CXX)

  install(TARGETS _circle_detection_cpp DESTINATION "${SKBUILD_PROJECT_NAME}")
  install(TARGETS _operations_cpp DESTINATION "${SKBUILD_PROJECT_NAME}/operations")
endif()
