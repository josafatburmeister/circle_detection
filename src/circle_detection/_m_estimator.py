""" M-Estimator circle detection method. """

__all__ = ["MEstimator"]

from typing import Optional, Union, cast
import numpy as np
import numpy.typing as npt

from ._circle_detection_cpp import (  # type: ignore[import-not-found] # pylint: disable = import-error
    detect_circles_m_estimator as detect_circles_m_estimator_cpp,
)
from ._circle_detector import CircleDetector


class MEstimator(CircleDetector):  # pylint: disable=too-many-instance-attributes
    r"""
    Detects circles in a set of 2D points using the M-estimator method proposed in `Garlipp, Tim, and Christine
    H. Müller. "Detection of Linear and Circular Shapes in Image Analysis." Computational Statistics & Data Analysis
    51.3 (2006): 1479-1490. <https://doi.org/10.1016/j.csda.2006.04.022>`__

    The input of the method a set of 2D points :math:`\{(x_1, y_1), ..., (x_N, y_N)\}`. Circles with different center
    positions and radii are generated as starting values for the circle detection process. The parameters of these
    initial circles are then optimized iteratively to align them with the input points.

    Given a circle with the center :math:`(a, b)` and a radius :math:`r`, the circle parameters are optimized
    by minimizing the following loss function (note that in the work by Garlipp and Müller, the function is formulated
    as a score function to be maximized and therefore has a positive instead of a negative sign):

    .. math::
        :nowrap:

        \begin{eqnarray}
            L(\begin{bmatrix} a, b, r \end{bmatrix}) = -\frac{1}{N} \sum_{i=1}^N \frac{1}{s} \rho
            \Biggl(
                \frac{\|\begin{bmatrix}x_i, y_i \end{bmatrix}^T - \begin{bmatrix} a, b \end{bmatrix}^T\| - r}{s}
            \Biggr)
        \end{eqnarray}

    Here, :math:`\rho` is a kernel function and :math:`s` is the kernel bandwidth. In this implementation, the standard
    Gaussian distribution with :math:`\mu = 0` and :math:`\sigma = 1` is used as the kernel function:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \rho(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}
        \end{eqnarray}

    To minimize the loss function :math:`L`,
    `Newton's method <https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization>`__ is used. In each
    optimization step, this method updates the parameters as follows:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \begin{bmatrix} a, b, r \end{bmatrix}_{t+1} = \begin{bmatrix} a, b, r \end{bmatrix}_t - s \cdot
            \nabla L([a, b, r]) \cdot \biggl[\nabla^2 L([a, b, r])\biggr]^{-1}
        \end{eqnarray}

    Here, :math:`s` is the step size. Since the Newton step requires to compute the inverse of the Hessian
    matrix of the loss function :math:`\bigl[\nabla^2 L([a, b, r])\bigr]^{-1}`, it can not be applied if the
    Hessian matrix is not invertible. Additionally, the Newton step only moves towards a local minimum if the
    determinant of the Hessian matrix is positive. Therefore, a simple gradient descent update step is used instead of
    the Newton step if the determinant of the Hessian matrix is not positive:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \begin{bmatrix} a, b, r \end{bmatrix}_{t+1} = \begin{bmatrix} a, b, r \end{bmatrix}_t - s \cdot
            \nabla L([a, b, r])
        \end{eqnarray}

    To determine a suitable step size :math:`s` for each step that results in a sufficient decrease of the loss
    function, the following rules are used:

    1. The initial step size :math:`s_0` is set to 1.
    2. If gradient descent is used in the optimization step, the step size is repeatedly increased by multiplying it
       with an acceleration factor :math:`\alpha > 1`, as long as increasing the step size results in a greater decrease
       of the loss function for the step. More formally, the step size update :math:`s_{k+1} = \alpha \cdot s_k` is
       repeated as long as the following condition is fullfilled:

       .. math::
           :nowrap:

           \begin{eqnarray}
           L(c + s_{k+1} \cdot d) < L(c + s_k \cdot d),
           \end{eqnarray}

       Here, :math:`d` is the step direction, :math:`c = [a, b, r]_{t}` are the circle's current parameters, and
       :math:`k` is the number of acceleration steps. No acceleration is applied for steps using Newton's method,
       as the Newton method itself includes an adjustment of the step size.

    3. If the gradient descent step size is not increased in (2) or Newton's method is used,
       the initial step size may be still too large to produce a sufficient decrease of the loss function. In this case,
       the initial step size is decreased until the step results in a sufficient decrease of the loss function. For this
       purpose, a `backtracking line-search <https://en.wikipedia.org/wiki/Backtracking_line_search>`__ according to
       `Armijo's rule <https://www.youtube.com/watch?v=Jxh2kqVz6lk>`__ is performed. Armijo's rule has two
       hyperparameters :math:`\beta \in (0, 1)` and :math:`\gamma \in (0,1)`. The step size is repeatedly decreased by
       multiplying it with the attenuation factor :math:`\beta` until the decrease of the loss function is at least a
       fraction :math:`\gamma` of the loss decrease expected based on a linear approximation of the loss function by its
       first-order Taylor polynomial. More formally, Armijo's rule repeats the step size update
       :math:`s_{k+1} = \beta \cdot s_k` until the following condition is fullfilled:

       .. math::
            :nowrap:

            \begin{eqnarray}
            L(c) - L(c + s_k \cdot d) \geq - \gamma \cdot s_k \cdot \nabla L(c) d^T
            \end{eqnarray}


    To be able to identify circles at different positions and of different sizes, the optimization is repeated with
    different starting values for the initial circle parameters. The starting values for the center coordinates
    :math:`(a, b)` are generated by placing points on a regular grid. The limits of the grid in x- and y-direction
    are defined by the parameters :code:`min_start_x`, :code:`max_start_x`, :code:`min_start_y`, and
    :code:`max_start_y`. The number of grid points is defined by the parametrs :code:`n_start_x` and :code:`n_start_y`.
    For each start position, :code:`n_start_radius` different starting values for the circle radius :math:`r` are
    tested, evenly covering the interval defined by :code:`min_start_radius` and :code:`max_start_radius`.

    To filter out circles for which the optimization does not converge, allowed value ranges are defined for the circle
    parameters by :code:`break_min_x`, :code:`break_max_x`, :code:`break_min_y`, :code:`break_max_y`,
    :code:`break_min_radius`, and :code:`break_max_radius`. If the parameters of a circle leave these value ranges
    during optimization, the optimization of the respective circle is terminated and the circle is discarded.

    Args:
        bandwidth: Kernel bandwidth.
        acceleration_factor: Acceleration factor :math:`\alpha` for increasing the step size. Defaults to 1.6.
        armijo_attenuation_factor: Attenuation factor :math:`\beta` for the backtracking line-search according to
            Armijo's rule. Defaults to 0.5.
        armijo_min_decrease_percentage: Hyperparameter :math:`\gamma` for the backtracking line-search according to
            Armijo's rule. Defaults to 0.1.
        min_step_size: Minimum step width. If the step size attenuation according to Armijo's rule results in a step
            size below this step size, the attenuation of the step size is terminated. Defaults to :math:`10^{-20}`.
        max_iterations: Maximum number of optimization iterations to run for each combination of starting values.
            Defaults to 1000.
        min_fitting_score: Minimum fitting score (equal to -1 :math:`\cdot` fitting loss) that a circle must have in
            order not to be discarded. Defaults to :math:`1`.

    Attributes:
        circles: After the :code:`self.detect()` method has been called, this attribute contains the parameters of the
            detected circles (in the following order: x-coordinate of the center, y-coordinate of the center, radius).
            If the :code:`self.detect()` method has not yet been called, this attribute is an empty array.
        fitting_scores: After the :code:`self.detect()` method has been called, this attribute contains the fitting
            scores of the detected circles (equal to -1 :math:`\cdot` fitting loss, higher means better). If the
            :code:`self.detect()` method has not yet been called, this attribute is an empty array.
        batch_lengths_circles: After the :code:`self.detect()` method has been called, this attribute contains the
            number of circles detected for each batch item (circles belonging to the same batch item are stored
            consecutively in :code:`self.circles`). If the :code:`self.detect()` method has not yet been
            called, this attribute is :code:`[0]`.

    Raises:
        ValueError: if :code:`acceleration_factor` is smaller than or equal to 1.
        ValueError: if :code:`armijo_attenuation_factor` is not within :math:`(0, 1)`.
        ValueError: if :code:`armijo_min_decrease_percentage` is not within :math:`(0, 1)`.
        ValueError: if :code:`min_step_size` is greater than :code:`break_min_change`.

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
        break_min_change: float = 1e-5,
        max_iterations: int = 1000,
        acceleration_factor: float = 1.6,
        armijo_attenuation_factor: float = 0.5,
        armijo_min_decrease_percentage: float = 0.1,
        min_step_size: float = 1e-20,
        min_fitting_score: float = 1,
    ):
        super().__init__()

        if acceleration_factor <= 1:
            raise ValueError("acceleration_factor must be > 1.")
        if armijo_attenuation_factor >= 1 or armijo_attenuation_factor <= 0:
            raise ValueError("armijo_attenuation_factor must be in (0, 1).")
        if armijo_min_decrease_percentage >= 1 or armijo_min_decrease_percentage <= 0:
            raise ValueError("armijo_min_decrease_percentage must be in (0, 1).")

        if min_step_size > break_min_change:
            raise ValueError("min_step_size should be smaller than break_min_change.")

        self._bandwidth = bandwidth
        self._break_min_change = break_min_change
        self._max_iterations = max_iterations
        self._acceleration_factor = acceleration_factor
        self._armijo_attenuation_factor = armijo_attenuation_factor
        self._armijo_min_decrease_percentage = armijo_min_decrease_percentage
        self._min_step_size = min_step_size
        self._min_fitting_score = min_fitting_score

    def detect(  # type: ignore[override] # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements, arguments-differ
        self,
        xy: npt.NDArray,
        *,
        batch_lengths: Optional[npt.NDArray[np.int64]] = None,
        min_start_x: Optional[Union[float, npt.NDArray]] = None,
        max_start_x: Optional[Union[float, npt.NDArray]] = None,
        n_start_x: int = 10,
        min_start_y: Optional[Union[float, npt.NDArray]] = None,
        max_start_y: Optional[Union[float, npt.NDArray]] = None,
        n_start_y: int = 10,
        min_start_radius: Optional[Union[float, npt.NDArray]] = None,
        max_start_radius: Optional[Union[float, npt.NDArray]] = None,
        n_start_radius: int = 10,
        break_min_x: Optional[Union[float, npt.NDArray]] = None,
        break_max_x: Optional[Union[float, npt.NDArray]] = None,
        break_min_y: Optional[Union[float, npt.NDArray]] = None,
        break_max_y: Optional[Union[float, npt.NDArray]] = None,
        break_min_radius: Optional[Union[float, npt.NDArray]] = None,
        break_max_radius: Optional[Union[float, npt.NDArray]] = None,
        num_workers: int = 1,
    ) -> None:
        r"""
        Executes the circle detection on the given input points. The results of the circle detection are stored in
        :code:`self.circles`, :code:`self.fitting_scores`, and :code:`self.batch_lengths_circles`.

        Args:
            xy: Coordinates of the set of 2D points in which to detect circles. If the :code:`xy` array has a row-major
                storage layout (`numpy's <https://numpy.org/doc/stable/dev/internals.html>`__ default), a copy of the
                array is created. To pass :code:`xy` by reference, :code:`xy` must be in column-major format.
            batch_lengths: Number of points in each point set of the input batch. For batch processing, it is
                expected that all points belonging to the same point set are stored consecutively in the :code:`xy`
                input array. For example, if the input is a batch of two point sets (i.e., two batch items) with
                :math:`N_1` points and :math:`N_2` points, then :code:`batch_lengths` should be set to
                :code:`[N_1, N_2]` and :code:`xy[:N_1]` should contain the points of the first point set and
                :code:`circles[N_1:]` the points of the second point set. If :code:`batch_lengths` is set to
                :code:`None`, it is assumed that the input points all belong to the same point set and batch processing
                is disabled. Defaults to :code:`None`.
            min_start_x: Lower limit of the start values for the x-coordinates of the circle centers. Can be either a
                scalar, an array of values (one per batch item), or :code:`None`. If a scalar is provided, the same
                value is used for all batch items. If set to :code:`None`, the minimum of the x-coordinates in of the
                points within each batch item is used as the default. Defaults to :code:`None`.
            max_start_x: Upper limit of the start values for the x-coordinates of the circle centers. Can be either a
                scalar, an array of values (one per batch item), or :code:`None`. If a scalar is provided, the same
                value is used for all batch items. If set to :code:`None`, the maximum of the x-coordinates in of the
                points within each batch item is used as the default. Defaults to :code:`None`.
            n_start_x: Number of start values for the x-coordinates of the circle centers. Defaults to 10.
            min_start_y: Lower limit of the start values for the y-coordinates of the circle centers. Can be either a
                scalar, an array of values (one per batch item), or :code:`None`. If a scalar is provided, the same
                value is used for all batch items. If set to :code:`None`, the minimum of the y-coordinates in of the
                points within each batch item is used as the default. Defaults to :code:`None`.
            max_start_y: Upper limit of the start values for the y-coordinates of the circle centers. Can be either a
                scalar, an array of values (one per batch item), or :code:`None`. If a scalar is provided, the same
                value is used for all batch items. If set to :code:`None`, the maximum of the y-coordinates in of the
                points within each batch item is used as the default. Defaults to :code:`None`.
            n_start_y: Number of start values for the y-coordinates of the circle centers. Defaults to 10.
            min_start_radius: Lower limit of the start values for the circle radii. Can be either a scalar, an array of
                values (one per batch item), or :code:`None`. If a scalar is provided, the same value is used for all
                batch items. If set to :code:`None`, :code:`0.1 * max_start_radius` is used as the default. Defaults to
                :code:`None`.
            max_start_radius: Upper limit of the start values for the circle radii. Can be either a scalar, an array of
                values (one per batch item), or :code:`None`. If a scalar is provided, the same value is used for all
                batch items. If set to :code:`None`, the axis-aligned bounding box of the points within each batch item
                is computed and the length of the longer side of the bounding box is used as the default. Defaults to
                :code:`None`.
            n_start_radius: Number of start values for the circle radii. Defaults to 10.
            break_min_x: Termination criterion for circle optimization. If the x-coordinate of a circle center becomes
                smaller than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`x_{start_{min}} - \max(0.1 \cdot (x_{start_{max}} - x_{start_{min}}), 2s)` is used
                as the default, where :math:`x_{start_{min}}` and :math:`x_{start_{max}}` are the minimum and the
                maximum start values for the x-coordinates of the circle centers, and :math:`s` is the bandwidth.
                Defaults to :code:`None`.
            break_max_x: Termination criterion for circle optimization. If the x-coordinate of a circle center becomes
                greater than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`x_{start_{max}} + \max(0.1 \cdot (x_{start_{max}} - x_{start_{min}}), 2s)` is used
                as the default, where :math:`x_{start_{min}}` and :math:`x_{start_{max}}` are the minimum and the
                maximum start values for the x-coordinates of the circle centers, and :math:`s` is the bandwidth.
                Defaults to :code:`None`.
            break_min_y: Termination criterion for circle optimization. If the y-coordinate of a circle center becomes
                smaller than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`y_{start_{min}} - \max(0.1 \cdot (y_{start_{max}} - y_{start_{min}}), 2s)` is used
                as the default, where :math:`y_{start_{min}}` and :math:`y_{start_{max}}` are the minimum and the
                maximum start values for the y-coordinates of the circle centers, and :math:`s` is the bandwidth.
                Defaults to :code:`None`.
            break_max_y: Termination criterion for circle optimization. If the y-coordinate of a circle center becomes
                greater than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`y_{start_{max}} + \max(0.1 \cdot (y_{start_{max}} - y_{start_{min}}), 2s)` is used
                as the default, where :math:`y_{start_{min}}` and :math:`y_{start_{max}}` are the minimum and the
                maximum start values for the y-coordinates of the circle centers, and :math:`s` is the bandwidth.
                Defaults to :code:`None`.
            break_min_radius: Termination criterion for circle optimization. If the radius of a circle center becomes
                smaller than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`r_{start_{min}} - \max(0.1 \cdot (r_{start_{max}} - r_{start_{min}}), 2s)` is used
                as the default, where :math:`r_{start_{min}}` and :math:`r_{start_{max}}` are the minimum and the
                maximum start radius, and :math:`s` is the bandwidth. Defaults to :code:`None`.
            break_max_radius: Termination criterion for circle optimization. If the radius of a circle center becomes
                greater than this value during optimization, the optimization of the respective circle is terminated and
                the respective circle is discarded. Can be either a scalar, an array of values (one per batch item), or
                :code:`None`. If a scalar is provided, the same value is used for all batch items. If set to
                :code:`None`, :math:`r_{start_{max}} + \max(0.1 \cdot (r_{start_{max}} - r_{start_{min}}), 2s)` is used
                as the default, where :math:`r_{start_{min}}` and :math:`r_{start_{max}}` are the minimum and the
                maximum start radius, and :math:`s` is the bandwidth. Defaults to :code:`None`.
            break_min_change: Termination criterion for circle optimization. If the updates of all circle parameters in
                an iteration are smaller than this threshold, the optimization of the respective circle is terminated.
                Defaults to :math:`10^{-5}`.
            num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are
                used. Defaults to 1.

        Raises:
            ValueError: if :code:`batch_lengths` is an empty array or the length of :code:`xy` is not equal to the sum
                of :code:`batch_lengths`.

            ValueError: If :code:`start_min_x`, :code:`start_max_x`, :code:`break_min_x`, :code:`break_max_x`,
                :code:`start_min_y`, :code:`start_max_y`, :code:`break_min_y`, :code:`break_max_y`,
                :code:`start_min_radius`, :code:`start_max_radius`, :code:`break_min_radius`, or
                :code:`break_max_radius` are arrays and their length is not equal to the batch size.

            ValueError: if :code:`min_start_x` is larger than :code:`max_start_x` for any batch item.
            ValueError: if :code:`min_start_x` is smaller than :code:`break_min_x` for any batch item.
            ValueError: if :code:`max_start_x` is smaller than :code:`break_max_x` for any batch item.

            ValueError: if :code:`min_start_y` is larger than :code:`max_start_y` for any batch item.
            ValueError: if :code:`min_start_y` is smaller than :code:`break_min_y` for any batch item.
            ValueError: if :code:`max_start_y` is smaller than :code:`break_max_y` for any batch item.

            ValueError: if :code:`min_start_radius` is larger than :code:`max_start_radius` for any batch item.
            ValueError: if :code:`min_start_radius` is smaller than :code:`break_min_radius` for any batch item.
            ValueError: if :code:`max_start_radius` is larger than :code:`break_max_radius` for any batch item.

            ValueError: if :code:`n_start_x` is not a positive number.
            ValueError: if :code:`n_start_y` is not a positive number.
            ValueError: if :code:`n_start_radius` is not a positive number.

        Shape:
            - :code:`xy`: :math:`(N, 2)`
            - :code:`batch_lengths`: :math:`(B)`
            - :code:`min_start_x`: scalar or array of shape :math:`(B)`
            - :code:`max_start_x`: scalar or array of shape :math:`(B)`
            - :code:`max_start_y`: scalar or array of shape :math:`(B)`
            - :code:`max_start_y`: scalar or array of shape :math:`(B)`
            - :code:`min_start_radius`: scalar or array of shape :math:`(B)`
            - :code:`max_start_radius`: scalar or array of shape :math:`(B)`
            - :code:`break_min_x`: scalar or array of shape :math:`(B)`
            - :code:`break_max_x`: scalar or array of shape :math:`(B)`
            - :code:`break_min_y`: scalar or array of shape :math:`(B)`
            - :code:`break_max_y`: scalar or array of shape :math:`(B)`
            - :code:`break_min_radius`: scalar or array of shape :math:`(B)`
            - :code:`break_max_radius`: scalar or array of shape :math:`(B)`

            | where
            |
            | :math:`B = \text{ batch size}`
            | :math:`N = \text{ number of points}`
        """

        if not xy.flags.f_contiguous:
            xy = xy.copy(order="F")  # ensure that the input array is in column-major

        if batch_lengths is None:
            batch_lengths = np.array([len(xy)], dtype=np.int64)
        num_batches = len(batch_lengths)
        batch_starts = np.cumsum(np.concatenate(([0], batch_lengths)))[:-1]
        batch_ends = np.cumsum(batch_lengths)

        if num_batches == 0:
            raise ValueError("batch_lengths must contain at least one entry.")

        if len(xy) != batch_lengths.sum():
            raise ValueError("The number of points must be equal to the sum of batch_lengths")

        if min_start_x is None:
            min_start_x = np.array(
                [
                    xy[batch_start:batch_end, 0].min() if batch_start < batch_end else 0
                    for (batch_start, batch_end) in zip(batch_starts, batch_ends)
                ],
                dtype=xy.dtype,
            )
        elif not isinstance(min_start_x, np.ndarray):
            min_start_x = np.full(num_batches, fill_value=min_start_x, dtype=xy.dtype)
        else:
            min_start_x = min_start_x.astype(xy.dtype)
        min_start_x = cast(npt.NDArray, min_start_x)

        if max_start_x is None:
            max_start_x = np.array(
                [
                    xy[batch_start:batch_end, 0].max() if batch_start < batch_end else 0
                    for (batch_start, batch_end) in zip(batch_starts, batch_ends)
                ],
                dtype=xy.dtype,
            )
        elif not isinstance(max_start_x, np.ndarray):
            max_start_x = np.full(num_batches, fill_value=max_start_x, dtype=xy.dtype)
        else:
            max_start_x = max_start_x.astype(xy.dtype)
        max_start_x = cast(npt.NDArray, max_start_x)

        if break_min_x is None:
            break_min_x = min_start_x - np.maximum(0.1 * (max_start_x - min_start_x), 2 * self._bandwidth)
        elif not isinstance(break_min_x, np.ndarray):
            break_min_x = np.full(num_batches, fill_value=break_min_x, dtype=xy.dtype)
        else:
            break_min_x = break_min_x.astype(xy.dtype)
        break_min_x = cast(npt.NDArray, break_min_x)

        if break_max_x is None:
            break_max_x = max_start_x + np.maximum(0.1 * (max_start_x - min_start_x), 2 * self._bandwidth)
        elif not isinstance(break_max_x, np.ndarray):
            break_max_x = np.full(num_batches, fill_value=break_max_x, dtype=xy.dtype)
        else:
            break_max_x = break_max_x.astype(xy.dtype)
        break_max_x = cast(npt.NDArray, break_max_x)

        if (min_start_x > max_start_x).any():
            raise ValueError("min_start_x must be smaller than or equal to max_start_x.")
        if (min_start_x < break_min_x).any():
            raise ValueError("min_start_x must be larger than or equal to min_break_x.")
        if (max_start_x > break_max_x).any():
            raise ValueError("max_start_x must be smaller than or equal to break_max_x.")

        if min_start_y is None:
            min_start_y = np.array(
                [
                    xy[batch_start:batch_end, 1].min() if batch_start < batch_end else 0
                    for (batch_start, batch_end) in zip(batch_starts, batch_ends)
                ],
                dtype=xy.dtype,
            )
        elif not isinstance(min_start_y, np.ndarray):
            min_start_y = np.full(num_batches, fill_value=min_start_y, dtype=xy.dtype)
        else:
            min_start_y = min_start_y.astype(xy.dtype)
        min_start_y = cast(npt.NDArray, min_start_y)

        if max_start_y is None:
            max_start_y = np.array(
                [
                    xy[batch_start:batch_end, 1].max() if batch_start < batch_end else 0
                    for (batch_start, batch_end) in zip(batch_starts, batch_ends)
                ],
                dtype=xy.dtype,
            )
        elif not isinstance(max_start_y, np.ndarray):
            max_start_y = np.full(num_batches, fill_value=max_start_y, dtype=xy.dtype)
        else:
            max_start_y = max_start_y.astype(xy.dtype)
        max_start_y = cast(npt.NDArray, max_start_y)

        if break_min_y is None:
            break_min_y = min_start_y - np.maximum(0.1 * (max_start_y - min_start_y), 2 * self._bandwidth)
        elif not isinstance(break_min_y, np.ndarray):
            break_min_y = np.full(num_batches, fill_value=break_min_y, dtype=xy.dtype)
        else:
            break_min_y = break_min_y.astype(xy.dtype)
        break_min_y = cast(npt.NDArray, break_min_y)

        if break_max_y is None:
            break_max_y = max_start_y + np.maximum(0.1 * (max_start_y - min_start_y), 2 * self._bandwidth)
        elif not isinstance(break_max_y, np.ndarray):
            break_max_y = np.full(num_batches, fill_value=break_max_y, dtype=xy.dtype)
        else:
            break_max_y = break_max_y.astype(xy.dtype)
        break_max_y = cast(npt.NDArray, break_max_y)

        if (min_start_y > max_start_y).any():
            raise ValueError("min_start_y must be smaller than or equal to max_start_y.")
        if (min_start_y < break_min_y).any():
            raise ValueError("min_start_y must be larger than or equal to min_break_y.")
        if (max_start_y > break_max_y).any():
            raise ValueError("max_start_y must be smaller than or equal to break_max_y.")

        if max_start_radius is None:
            max_start_radius = np.array(
                [
                    (
                        (xy[batch_start:batch_end].max(axis=0) - xy[batch_start:batch_end].min(axis=0)).max()
                        if batch_start < batch_end
                        else 0.1
                    )
                    for (batch_start, batch_end) in zip(batch_starts, batch_ends)
                ],
                dtype=xy.dtype,
            )
        elif not isinstance(max_start_radius, np.ndarray):
            max_start_radius = np.full(num_batches, fill_value=max_start_radius, dtype=xy.dtype)
        else:
            max_start_radius = max_start_radius.astype(xy.dtype)
        max_start_radius = cast(npt.NDArray, max_start_radius)

        if min_start_radius is None:
            min_start_radius = 0.1 * max_start_radius
        elif not isinstance(min_start_radius, np.ndarray):
            min_start_radius = np.full(num_batches, fill_value=min_start_radius, dtype=xy.dtype)
        else:
            min_start_radius = min_start_radius.astype(xy.dtype)
        min_start_radius = cast(npt.NDArray, min_start_radius)

        if break_min_radius is None:
            break_min_radius = min_start_radius - np.maximum(
                0.1 * (max_start_radius - min_start_radius), 2 * self._bandwidth
            )
        elif not isinstance(break_min_radius, np.ndarray):
            break_min_radius = np.full(num_batches, fill_value=break_min_radius, dtype=xy.dtype)
        else:
            break_min_radius = break_min_radius.astype(xy.dtype)
        break_min_radius = cast(npt.NDArray, break_min_radius)

        if break_max_radius is None:
            break_max_radius = max_start_radius + np.maximum(
                0.1 * (max_start_radius - min_start_radius), 2 * self._bandwidth
            )
        elif not isinstance(break_max_radius, np.ndarray):
            break_max_radius = np.full(num_batches, fill_value=break_max_radius, dtype=xy.dtype)
        else:
            break_max_radius = break_max_radius.astype(xy.dtype)
        break_max_radius = cast(npt.NDArray, break_max_radius)

        if (min_start_radius < 0).any():
            raise ValueError("min_start_radius must be larger than zero.")

        if (min_start_radius > max_start_radius).any():
            raise ValueError("min_start_radius must be smaller than or equal to max_start_radius.")
        if (min_start_radius < break_min_radius).any():
            raise ValueError("min_start_radius must be larger than or equal to break_min_radius.")
        if (max_start_radius > break_max_radius).any():
            raise ValueError("max_start_radius must be smaller than or equal to break_max_radius.")

        if n_start_x <= 0:
            raise ValueError("n_start_x must be a positive number.")
        if n_start_y <= 0:
            raise ValueError("n_start_y must be a positive number.")
        if n_start_radius <= 0:
            raise ValueError("n_start_radius must be a positive number.")

        break_min_radius = np.maximum(break_min_radius, 0)

        self.circles, self.fitting_scores, self.batch_lengths_circles = detect_circles_m_estimator_cpp(
            xy,
            batch_lengths,
            float(self._bandwidth),
            min_start_x,
            max_start_x,
            int(n_start_x),
            min_start_y,
            max_start_y,
            int(n_start_y),
            min_start_radius,
            max_start_radius,
            int(n_start_radius),
            break_min_x,
            break_max_x,
            break_min_y,
            break_max_y,
            break_min_radius,
            break_max_radius,
            float(self._break_min_change),
            int(self._max_iterations),
            float(self._acceleration_factor),
            float(self._armijo_attenuation_factor),
            float(self._armijo_min_decrease_percentage),
            float(self._min_step_size),
            float(self._min_fitting_score),
            int(num_workers),
        )

        self._xy = xy
        self._batch_lengths_xy = batch_lengths
        self._has_detected_circles = True
