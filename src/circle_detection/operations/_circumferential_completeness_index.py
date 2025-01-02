""" Operations to calculate the circumferential completeness index and to filter circles based on this metric. """

__all__ = [
    "circumferential_completeness_index",
    "filter_circumferential_completeness_index",
]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from circle_detection.operations._operations_cpp import (  # type: ignore[import-not-found] # pylint: disable=import-error, no-name-in-module
    circumferential_completeness_index as circumferential_completeness_index_cpp,
    filter_circumferential_completeness_index as filter_circumferential_completeness_index_cpp,
)


def circumferential_completeness_index(
    circles: npt.NDArray[np.float64],
    xy: npt.NDArray[np.float64],
    num_regions: int,
    max_dist: Optional[float] = None,
    batch_lengths_circles: Optional[npt.NDArray[np.int64]] = None,
    batch_lengths_xy: Optional[npt.NDArray[np.int64]] = None,
    num_workers: int = 1,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the circumferential completeness indices of the specified circles. The circumferential completeness index
    is a metric that measures how well a circle fitted to a set of points is covered by points. It was proposed in
    `Krisanski, Sean, Mohammad Sadegh Taskhiri, and Paul Turner. "Enhancing Methods for Under-canopy Unmanned Aircraft \
    System Based Photogrammetry in Complex Forests for Tree Diameter Measurement." Remote Sensing 12.10 (2020): 1652. \
    <https://doi.org/10.3390/rs12101652>`__ To calculate the circumference completeness index of a circle, the circle is
    divided into :code:`num_regions` angular regions. An angular region is considered complete if it contains at least
    one point whose distance to the circle outline is equal to or less than :code:`max_dist`. The circumferential
    completeness index is then defined as the proportion of angular regions that are complete. This method supports
    batch processing, i.e. separate sets of circles (i.e., different batch items) can be processed in parallel. For this
    purpose, :code:`batch_lengths_circles` and :code:`batch_lengths_xy` must be set to specify which circle / which
    point belongs to which set.

    Args:
        circles: Parameters of the circles for which to compute the circumferential completeness indices. Each circle
            must be defined by three parameters in the following order: x-coordinate of the center, y-coordinate of the
            center, radius.
        xy: Coordinates of the set of 2D points to which the circles were fitted.
        num_regions: Number of angular regions.
        max_dist: Maximum distance a point can have to the circle outline to be counted as part of the circle. If set to
            :code:`None`, points are counted as part of the circle if their distance to the circle is center is in the
            interval :math:`[0.7 \cdot r, 1.3 \cdot r]` where :math:`r` is the circle radius. Defaults to :code:`None`.
        batch_lengths_circles: Number of circles in each item of the input batch. For batch processing, it is
            expected that all circles belonging to the same batch item are stored consecutively in the :code:`circles`
            input array. For example, if a batch comprises two batch items with :math:`N_1` circles and :math:`N_2`
            circles, then :code:`batch_lengths_circles` should be set to :code:`[N_1, N_2]` and :code:`circles[:N_1]`
            should contain the circles of the first batch item and :code:`circles[N_1:]` the circles of the second batch
            item. If :code:`batch_lengths_circles` is set to :code:`None`, it is assumed that the input circles
            belong to a single batch item and batch processing is disabled. Defaults to :code:`None`.
        batch_lengths_xy: Number of points in each item of the input batch. For batch processing, it is
            expected that all points belonging to the same batch item are stored consecutively in the :code:`xy`
            input array. If :code:`batch_lengths_xy` is set to :code:`None`, it is assumed that the input points
            belong to a single batch item and batch processing is disabled. Defaults to :code:`None`.
        num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are used.
            Defaults to 1.

    Returns:
        Circumferential completeness indices of the circles.

    Raises:
        ValueError: If the length of :code:`circles` is not equal to the sum of :code:`batch_lengths_circles`, if
            the length of :code:`xy` is not equal to the sum of :code:`batch_lengths_xy`, if
            :code:`batch_lengths_circles` is :code:`None` and :code:`batch_lengths_xy` not (or vice versa), or if
            :code:`batch_lengths_circles` and :code:`batch_lengths_xy` have different lengths.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`xy`: :math:`(N, 2)`
        - :code:`batch_lengths_circles`: :math:`(B)`
        - :code:`batch_lengths_circles`: :math:`(B)`
        - Output: :math:`(C)`

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of circles}`
        | :math:`N = \text{ number of points}`
    """

    if batch_lengths_circles is None and batch_lengths_xy is not None:
        raise ValueError("batch_lengths_circles must not be None if batch_lengths_xy is specified.")
    if batch_lengths_xy is None and batch_lengths_circles is not None:
        raise ValueError("batch_lengths_xy must not be None if batch_lengths_circles is specified.")

    if batch_lengths_circles is None:
        batch_lengths_circles = np.array([len(circles)], dtype=np.int64)
    if batch_lengths_xy is None:
        batch_lengths_xy = np.array([len(xy)], dtype=np.int64)
    if max_dist is None:
        max_dist = -1

    return circumferential_completeness_index_cpp(
        circles, xy, batch_lengths_circles, batch_lengths_xy, int(num_regions), float(max_dist), int(num_workers)
    )


def filter_circumferential_completeness_index(
    circles: npt.NDArray[np.float64],
    xy: npt.NDArray[np.float64],
    num_regions: int,
    min_circumferential_completeness_index: float,
    max_dist: Optional[float] = None,
    batch_lengths_circles: Optional[npt.NDArray[np.int64]] = None,
    batch_lengths_xy: Optional[npt.NDArray[np.int64]] = None,
    num_workers: int = 1,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Filters out the circles whose circumferential completeness index is below the specified minimum circumferential
    completeness index. This method supports batch processing, i.e. separate sets of circles (i.e., different batch
    items) can be filtered in parallel. For this purpose, :code:`batch_lengths_circles` and :code:`batch_lengths_xy`
    must be set to specify which circle / which point belongs to which set.

    Args:
        circles: Parameters of the circles for which to compute the circumferential completeness indices. Each circle
            must be defined by three parameters in the following order: x-coordinate of the center, y-coordinate of the
            center, radius.
        xy: Coordinates of the set of 2D points to which the circles were fitted.
        num_regions: Number of angular regions.
        min_circumferential_completeness_index: Minimum circumferential index a point must have to not be discarded.
        max_dist: Maximum distance a point can have to the circle outline to be counted as part of the circle. If set to
            :code:`None`, points are counted as part of the circle if their distance to the circle is center is in the
            interval :math:`[0.7 \cdot r, 1.3 \cdot r]` where :math:`r` is the circle radius. Defaults to :code:`None`.
        batch_lengths_circles: Number of circles in each item of the input batch. For batch processing, it is
            expected that all circles belonging to the same batch item are stored consecutively in the :code:`circles`
            input array. For example, if a batch comprises two batch items with :math:`N_1` circles and :math:`N_2`
            circles, then :code:`batch_lengths_circles` should be set to :code:`[N_1, N_2]` and :code:`circles[:N_1]`
            should contain the circles of the first batch item and :code:`circles[N_1:]` the circles of the second batch
            item. If :code:`batch_lengths_circles` is set to :code:`None`, it is assumed that the input circles
            belong to a single batch item and batch processing is disabled. Defaults to :code:`None`.
        batch_lengths_xy: Number of points in each item of the input batch. For batch processing, it is
            expected that all points belonging to the same batch item are stored consecutively in the :code:`xy`
            input array. If :code:`batch_lengths_xy` is set to :code:`None`, it is assumed that the input points
            belong to a single batch item and batch processing is disabled. Defaults to :code:`None`.
        num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are used.
            Defaults to 1.

    Returns:
        : Tuple of three arrays. The first contains the parameters of the circles remaining after filtering.
        The second contains the number of circles in each item of the output batch. The third contains the indices of
        the selected circles in the input array.

    Raises:
        ValueError: If the length of :code:`circles` is not equal to the sum of :code:`batch_lengths_circles`, if
            the length of :code:`xy` is not equal to the sum of :code:`batch_lengths_xy`, if
            :code:`batch_lengths_circles` is :code:`None` and :code:`batch_lengths_xy` not (or vice versa), or if
            :code:`batch_lengths_circles` and :code:`batch_lengths_xy` have different lengths.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`xy`: :math:`(N, 2)`
        - Output: Tuple of three arrays. The first has shape :math:`(C', 3)` and the second shape :math:`(B)`, and the
          third shape :math:`(C')`.

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of circles before the filtering}`
        | :math:`C' = \text{ number of circles after the filtering}`
        | :math:`N = \text{ number of points}`
    """
    if batch_lengths_circles is None and batch_lengths_xy is not None:
        raise ValueError("batch_lengths_circles must not be None if batch_lengths_xy is specified.")
    if batch_lengths_xy is None and batch_lengths_circles is not None:
        raise ValueError("batch_lengths_xy must not be None if batch_lengths_circles is specified.")

    if batch_lengths_circles is None:
        batch_lengths_circles = np.array([len(circles)], dtype=np.int64)
    if batch_lengths_xy is None:
        batch_lengths_xy = np.array([len(xy)], dtype=np.int64)
    if max_dist is None:
        max_dist = -1

    return filter_circumferential_completeness_index_cpp(
        circles,
        xy,
        batch_lengths_circles,
        batch_lengths_xy,
        int(num_regions),
        float(max_dist),
        float(min_circumferential_completeness_index),
        int(num_workers),
    )
