# circle_detection

### A Python Package for Detecting Circles in 2D Point Sets.

![pypi-image](https://badge.fury.io/py/circle-detection.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/josafatburmeister/circle_detection/actions/workflows/code-quality-main.yml/badge.svg)](https://github.com/josafatburmeister/circle_detection/actions/workflows/code-quality-main.yml)
[![coverage](https://codecov.io/gh/josafatburmeister/circle_detection/branch/main/graph/badge.svg)](https://codecov.io/github/josafatburmeister/circle_detection?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/circle_detection)

The package allows to detect circles in a set of 2D points. Currently, the package implements the following circle
detection methods:

- The M-estimator method proposed in [Garlipp, Tim, and Christine H. Müller. "Detection of Linear and Circular Shapes in Image Analysis." Computational Statistics & Data Analysis 51.3 (2006): 1479-1490.](<https://doi.org/10.1016/j.csda.2006.04.022>)
- The [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) method based on least-squares circle fitting.

### Get started

The package can be installed via pip:

```bash
python -m pip install circle-detection
```

The package provides ```MEstimator``` and ```Ransac``` classes, which can be used as follows:

```python
from circle_detection import MEstimator, Ransac
import numpy as np

xy = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)

m_estimator = MEstimator(bandwidth=0.05)
m_estimator.detect(xy)
m_estimator.filter(max_circles=1)

print("Circles detected by M-Estimator method:", np.round(m_estimator.circles, 2))
print("Fitting scores:", m_estimator.fitting_scores)

ransac = Ransac(bandwidth=0.05)
ransac.detect(xy)
ransac.filter(max_circles=1)

print("Circles detected by RANSAC method:", np.round(ransac.circles, 2))
print("Fitting scores:", ransac.fitting_scores)

# Retrieve parameters of the detected circles
if len(ransac.circles) > 0:
    circle_center_x, circle_center_y, circle_radius = ransac.circles[0]
```

The package also supports batch processing, i.e. the parallel detection of circles in separate sets of points. For batch
processing, the points of all input point sets must be stored in a flat array. Points that belong to the same point set
must be stored consecutively. The number of points per point set must then be specified using the `batch_lengths`
parameter:

```python
from circle_detection import Ransac
import numpy as np

xy = np.array(
    [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [0, 1],
        [2, 1],
        [1, 0],
        [1, 2],
        [1 + np.sqrt(2), 1 + np.sqrt(2)],
    ],
    dtype=np.float64,
)
batch_lengths = np.array([4, 5], dtype=np.int64)

circle_detector = Ransac(bandwidth=0.05)
circle_detector.detect(xy, batch_lengths=batch_lengths)
circle_detector.filter(max_circles=1)

print("Circles:", np.round(circle_detector.circles, 2))
print("Fitting scores:", circle_detector.fitting_scores)
print("Number of circles detected in each point set:", circle_detector.batch_lengths_circles)
```

### Package Documentation

The package documentation is available [here](https://josafatburmeister.github.io/circle_detection/stable).
