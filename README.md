# circle_detection

### A Python Package for Detecting Circles in 2D Point Sets.

![pypi-image](https://badge.fury.io/py/circle-detection.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/josafatburmeister/circle_detection/actions/workflows/code-quality-main.yml/badge.svg)](https://github.com/josafatburmeister/circle_detection/actions/workflows/code-quality-main.yml)
[![coverage](https://codecov.io/gh/josafatburmeister/circle_detection/branch/main/graph/badge.svg)](https://codecov.io/github/josafatburmeister/circle_detection?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/circle_detection)

The package allows to detect circles in a set of 2D points using the M-estimator method proposed in [Garlipp, Tim, and Christine H. Müller. "Detection of Linear and Circular Shapes in Image Analysis." Computational Statistics & Data Analysis 51.3 (2006): 1479-1490.](<https://doi.org/10.1016/j.csda.2006.04.022>)

### Get started

The package can be installed via pip:

```bash
python -m pip install circle-detection
```

The package provides a ```CircleDetection``` class, which can be used as follows:

```python
from circle_detection import CircleDetection
import numpy as np

xy = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)

circle_detector = CircleDetection(bandwidth=0.05)
circle_detector.detect(xy)
circle_detector.filter(max_circles=1)

print("circles:", circle_detector.circles)
print("fitting losses:", circle_detector.fitting_losses)

if len(circle_detector.circles) > 0:
    circle_center_x, circle_center_y, circle_radius = circle_detector.circles[0]
```

The package also supports batch processing, i.e. the parallel detection of circles in separate sets of points. For batch
processing, the points of all input point sets must be stored in a flat array. Points that belong to the same point set
must be stored consecutively. The number of points per point set must then be specified using the `batch_lengths`
parameter:

```python
from circle_detection import CircleDetection
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

circle_detector = CircleDetection(bandwidth=0.05)
circle_detector.detect(xy, batch_lengths=batch_lengths)
circle_detector.filter(max_circles=1)

print("circles:", circle_detector.circles)
print("fitting losses:", circle_detector.fitting_losses)
print("number of circles detected in each point set:" circle_detector.batch_lengths_circles)
```

### Package Documentation

The package documentation is available [here](https://josafatburmeister.github.io/circle_detection/stable).
