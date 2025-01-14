""" RANSAC circle detection method. """

__all__ = ["Ransac"]

from typing import Optional, Union, cast

import numpy as np
import numpy.typing as npt

from ._circle_detection_cpp import (  # type: ignore[import-not-found] # pylint: disable = import-error
    detect_circles_ransac as detect_circles_ransac_cpp,
)
from ._circle_detector import CircleDetector


class Ransac(CircleDetector):
    r"""
    Detects circles in a set of 2D points using the
    `Random sample consensus (RANSAC) <https://en.wikipedia.org/wiki/Random_sample_consensus>`__ algorithm. The RANSAC
    algorithm consists of the following steps that are repeated for a certain number of iterations:

    1. A random, minimal subset is sampled from the original set of input points. This subset is called the hypothetical
       inliers. The cardinality of the subset must be sufficient to determine the circle parameters (i.e., it must
       contain at least three points).
    2. A circle is fitted to the set of hypothetical inliers.
    3. All points from the original set of input points are then tested against the fitted circle. All points whose
       distance to the outline of the fitted circle is above some error threshold are considered outliers. The remaining
       points that fit the estimated circle well according to the error threshold are considered the consensus set.
    4. The circle fit is improved by re-estimating it by using all the members of the consensus set.
    5. The quality of fit of the re-estimated circle is evaluated using a scoring function. The circle is accepted if
       its fitting score is above a certain threshold, the cardinality of its consensus set is sufficiently large and
       its parameters lie within a value range specified by the user.

    For the circle fitting in step 2, a least-squares estimator is used, which minimizes the squared distances of the
    points to the circle outline. The implementation of the least-squares estimator is based on that from the
    `scikit-image <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.CircleModel>`__
    package.

    To measure the quality of fit in step 5, the scoring function proposed in `Garlipp, Tim, and Christine H. Müller. \
    "Detection of Linear and Circular Shapes in Image Analysis." Computational Statistics & Data Analysis 51.3 (2006): \
    1479-1490. <https://doi.org/10.1016/j.csda.2006.04.022>`__ is used:

    .. math::
        :nowrap:

        \begin{eqnarray}
            L(\begin{bmatrix} a, b, r \end{bmatrix}) = -\frac{1}{N} \sum_{i=1}^N \frac{1}{s} \rho
            \Biggl(
                \frac{\|\begin{bmatrix}x_i, y_i \end{bmatrix}^T - \begin{bmatrix} a, b \end{bmatrix}^T\| - r}{s}
            \Biggr)
        \end{eqnarray}

    Here, :math:`\{(x_1, y_1), ..., (x_N, y_N)\}` is the set of input points and :math:`(a, b, r)` are the circle
    parameters, :math:`\rho` is a kernel function, and :math:`s` is the kernel bandwidth. The standard Gaussian
    distribution is used as the kernel function and the kernel bandwidth is set to the RANSAC error threshold.

    Args:
        bandwidth: Error threshold defining which points are considered outliers.
        iterations: How often the RANSAC procedure is repeated. Increasing the number of iterations increases the
            probability of obtaining at least one fitting circle. Defaults to 1000.
        num_samples: Number of hypothetical inliers to sample in each iteration. Defaults to 3.
        min_concensus_points: Minimum size the consensus set of a circle must have in order to accept the circle.
            Defaults to 3.
        min_fitting_score: Minimum fitting score a circle must have in order to be accepted. Defaults to :code:`100`.

    Raises:
        ValueError: If :code:`num_samples` or :code:`min_concensus_points` are smaller than 3.

    Attributes:
        circles: After the :code:`self.detect()` method has been called, this attribute contains the parameters of the
            detected circles (in the following order: x-coordinate of the center, y-coordinate of the center, radius).
            If the :code:`self.detect()` method has not yet been called, this attribute is an empty array.
        fitting_scores: After the :code:`self.detect()` method has been called, this attribute contains the fitting
            scores of the detected circles (higher means better). If the :code:`self.detect()`
            method has not yet been called, this attribute is an empty array.
        batch_lengths_circles: After the :code:`self.detect()` method has been called, this attribute contains the
            number of circles detected for each batch item (circles belonging to the same batch item are stored
            consecutively in :code:`self.circles`). If the :code:`self.detect()` method has not yet been
            called, this attribute is :code:`[0]`.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`fitting_scores`: :math:`(C)`
        - :code:`batch_lengths_circles`: :math:`(B)`

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of detected circles}`
    """

    def __init__(
        self,
        bandwidth: float,
        *,
        iterations: int = 1000,
        num_samples: int = 3,
        min_concensus_points: int = 3,
        min_fitting_score: float = 100,
    ):
        super().__init__()

        if num_samples < 3:
            raise ValueError("The required number of hypothetical inlier points must be at least 3.")

        if min_concensus_points < 3:
            raise ValueError("The required number of consensus points must be at least 3.")

        self._bandwidth = bandwidth
        self._iterations = iterations
        self._num_samples = num_samples
        self._min_concensus_points = min_concensus_points
        self._min_fitting_score = min_fitting_score

    def detect(  # type: ignore[override] # pylint: disable=arguments-differ, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        self,
        xy: npt.NDArray,
        *,
        batch_lengths: Optional[npt.NDArray[np.int64]] = None,
        break_min_x: Optional[Union[float, npt.NDArray]] = None,
        break_max_x: Optional[Union[float, npt.NDArray]] = None,
        break_min_y: Optional[Union[float, npt.NDArray]] = None,
        break_max_y: Optional[Union[float, npt.NDArray]] = None,
        break_min_radius: Optional[Union[float, npt.NDArray]] = None,
        break_max_radius: Optional[Union[float, npt.NDArray]] = None,
        num_workers: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Executes the circle detection on the given input points. The results of the circle detection are stored in
        :code:`self.circles`, :code:`self.fitting_scores`, and :code:`self.batch_lengths_circles`.

        Args:
            xy: Coordinates of the set of 2D points in which to detect circles. If the :code:`xy` array has a row-major
                storage layout (`numpy's <https://numpy.org/doc/stable/dev/internals.html>`__ default) a copy of the
                array is created. To pass :code:`xy` by reference, :code:`xy` must be in column-major format.
            batch_lengths: Number of points in each point set of the input batch. For batch processing, it is
                expected that all points belonging to the same point set are stored consecutively in the :code:`xy`
                input array. For example, if the input is a batch of two point sets (i.e., two batch items) with
                :math:`N_1` points and :math:`N_2` points, then :code:`batch_lengths` should be set to
                :code:`[N_1, N_2]` and :code:`xy[:N_1]` should contain the points of the first point set and
                :code:`circles[N_1:]` the points of the second point set. If :code:`batch_lengths` is set to
                :code:`None`, it is assumed that the input points all belong to the same point set and batch processing
                is disabled. Defaults to :code:`None`.
            break_min_x: Rejection criterion for circle fitting. If the x-coordinate of a circle center is smaller than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one per
                batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items.
                If set to :code:`None`, :math:`(x_{min} - 2s)` is used as the default, where :math:`x_{min}` is the
                minimum of the x-coordinates of the points within a batch item and :math:`s` is the bandwidth. Defaults
                to :code:`None`.
            break_max_x: Rejection criterion for circle fitting. If the x-coordinate of a circle center is larger than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one
                per batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items.
                If set to :code:`None`, :math:`(x_{max} + 2s)` is used as the default, where :math:`x_{max}` is the
                maximum of the x-coordinates of the points within a batch item and :math:`s` is the bandwidth. Defaults
                to :code:`None`.
            break_min_y: Rejection criterion for circle fitting. If the y-coordinate of a circle center is smaller than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one per
                batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items.
                If set to :code:`None`, :math:`(y_{min} - 2s)` is used as the default, where :math:`y_{min}` is the
                minimum of the y-coordinates of the points within a batch item and :math:`s` is the bandwidth. Defaults
                to :code:`None`.
            break_max_y: Rejection criterion for circle fitting. If the x-coordinate of a circle center is larger than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one
                per batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items.
                If set to :code:`None`, :math:`(y_{max} + 2s)` is used as the default, where :math:`x_{min}` is the
                maximum of the y-coordinates of the points within a batch item and :math:`s` is the bandwidth. Defaults
                to :code:`None`.
            break_min_radius: Rejection criterion for circle fitting. If the radius of a circle center is smaller than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one per
                batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items. If
                set to :code:`None`, zero is used as the default. Defaults to :code:`None`.
            break_max_radius:  Rejection criterion for circle fitting. If the radius of a circle center is larger than
                this value, the respective circle is discarded. Can be either a scalar, an array of values (one per
                batch item), or :code:`None`. If a scalar is provided, the same value is used for all batch items. If
                set to :code:`None`, :math:`max(x_{max} - x_{min}, y_{max} - y_{min}) + 2s` is used as the default,
                where :math:`s` is the bandwidth and :math:`x_{min}`, :math:`x_{max}`, :math:`y_{min}`, and
                :math:`y_{max}` are the minimum and the maximum of the x- and y-coordinates of the points within a batch
                item, respectively. Defaults to :code:`None`.
            num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are
                used. Defaults to 1.
            seed: Random seed. If set to :code:`None`, the random processes are not seeded. Defaults to :code:`None`.

        Raises:
            ValueError: if :code:`batch_lengths` is an empty array or the length of :code:`xy` is not equal to the sum
                of :code:`batch_lengths`.
            ValueError: If :code:`break_min_x`, :code:`break_max_x`, :code:`break_min_y`, :code:`break_max_y`,
                :code:`break_min_radius`, or :code:`break_max_radius` are arrays and their length is not equal to the
                batch size.

            ValueError: If :code:`break_min_x` is larger than :code:`break_max_x`.
            ValueError: If :code:`break_min_y` is larger than :code:`break_max_y`.
            ValueError: If :code:`break_min_radius` is larger than :code:`break_max_radius`.
        """

        if not xy.flags.f_contiguous:
            xy = xy.copy(order="F")  # ensure that the input array is in column-major format

        if batch_lengths is None:
            batch_lengths = np.array([len(xy)], dtype=np.int64)
        num_batches = len(batch_lengths)
        batch_starts = np.cumsum(np.concatenate(([0], batch_lengths)))[:-1]
        batch_ends = np.cumsum(batch_lengths)

        if num_batches == 0:
            raise ValueError("batch_lengths must contain at least one entry.")

        min_start_x = np.array(
            [
                xy[batch_start:batch_end, 0].min() if batch_start < batch_end else 0
                for (batch_start, batch_end) in zip(batch_starts, batch_ends)
            ],
            dtype=xy.dtype,
        )

        max_start_x = np.array(
            [
                xy[batch_start:batch_end, 0].max() if batch_start < batch_end else 0
                for (batch_start, batch_end) in zip(batch_starts, batch_ends)
            ],
            dtype=xy.dtype,
        )

        if break_min_x is None:
            break_min_x = min_start_x - 2 * self._bandwidth
        elif not isinstance(break_min_x, np.ndarray):
            break_min_x = np.full(num_batches, fill_value=break_min_x, dtype=xy.dtype)
        break_min_x = cast(npt.NDArray, break_min_x)

        if break_max_x is None:
            break_max_x = max_start_x + 2 * self._bandwidth
        elif not isinstance(break_max_x, np.ndarray):
            break_max_x = np.full(num_batches, fill_value=break_max_x, dtype=xy.dtype)
        break_max_x = cast(npt.NDArray, break_max_x)

        if (break_min_x >= break_max_x).any():
            raise ValueError("break_min_x must be smaller than break_max_x.")

        min_start_y = np.array(
            [
                xy[batch_start:batch_end, 1].min() if batch_start < batch_end else 0
                for (batch_start, batch_end) in zip(batch_starts, batch_ends)
            ],
            dtype=xy.dtype,
        )

        max_start_y = np.array(
            [
                xy[batch_start:batch_end, 1].max() if batch_start < batch_end else 0
                for (batch_start, batch_end) in zip(batch_starts, batch_ends)
            ],
            dtype=xy.dtype,
        )

        if break_min_y is None:
            break_min_y = min_start_y - 2 * self._bandwidth
        elif not isinstance(break_min_y, np.ndarray):
            break_min_y = np.full(num_batches, fill_value=break_min_y, dtype=xy.dtype)
        break_min_y = cast(npt.NDArray, break_min_y)

        if break_max_y is None:
            break_max_y = max_start_y + 2 * self._bandwidth
        elif not isinstance(break_max_y, np.ndarray):
            break_max_y = np.full(num_batches, fill_value=break_max_y, dtype=xy.dtype)
        break_max_y = cast(npt.NDArray, break_max_y)

        if (break_min_y > break_max_y).any():
            raise ValueError("break_min_y must be smaller than break_max_y.")

        max_start_radius = np.array(
            [
                (
                    (xy[batch_start:batch_end].max(axis=0) - xy[batch_start:batch_end].min(axis=0)).max()
                    if batch_start < batch_end
                    else 0.1
                )
                for (batch_start, batch_end) in zip(batch_starts, batch_ends)
            ]
        )

        if break_min_radius is None:
            break_min_radius = np.zeros(num_batches, dtype=xy.dtype)
        elif not isinstance(break_min_radius, np.ndarray):
            break_min_radius = np.full(num_batches, fill_value=break_min_radius, dtype=xy.dtype)
        break_min_radius = cast(npt.NDArray, break_min_radius)

        if break_max_radius is None:
            break_max_radius = max_start_radius + 2 * self._bandwidth
        elif not isinstance(break_max_radius, np.ndarray):
            break_max_radius = np.full(num_batches, fill_value=break_max_radius, dtype=xy.dtype)
        break_max_radius = cast(npt.NDArray, break_max_radius)

        if (break_min_radius > break_max_radius).any():
            raise ValueError("break_min_radius must be smaller than break_max_radius.")

        break_min_radius = np.maximum(break_min_radius, 0)

        if seed is None:
            seed = -1

        self.circles, self.fitting_scores, self.batch_lengths_circles = detect_circles_ransac_cpp(
            xy,
            batch_lengths,
            break_min_x,
            break_max_x,
            break_min_y,
            break_max_y,
            break_min_radius,
            break_max_radius,
            float(self._bandwidth),
            int(self._iterations),
            int(self._num_samples),
            int(self._min_concensus_points),
            float(self._min_fitting_score),
            int(num_workers),
            int(seed),
        )

        self._xy = xy
        self._batch_lengths_xy = batch_lengths
        self._has_detected_circles = True
