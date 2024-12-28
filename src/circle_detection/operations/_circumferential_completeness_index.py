""" Operations to calculate the circumferential completeness index and to filter circles based on this metric. """

__all__ = ["circumferential_completeness_index", "filter_circumferential_completeness_index"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def circumferential_completeness_index(
    circles: npt.NDArray[np.float64], xy: npt.NDArray[np.float64], num_regions: int, max_dist: Optional[float] = None
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the circumferential completeness indices of the specified circles. The circumferential completeness index
    is a metric that measures how well a circle fitted to a set of points is covered by points. It was proposed in
    `Krisanski, Sean, Mohammad Sadegh Taskhiri, and Paul Turner. "Enhancing Methods for Under-canopy Unmanned Aircraft \
    System Based Photogrammetry in Complex Forests for Tree Diameter Measurement." Remote Sensing 12.10 (2020): 1652. \
    <https://doi.org/10.3390/rs12101652>`__ To calculate the circumference completeness index of a circle, the circle is
    divided into :code:`num_regions` angular regions. An angular region is considered complete if it contains at least
    one point whose distance to the circle outline is equal to or less than :code:`max_dist`. The circumferential
    completeness index is then defined as the proportion of angular regions that are complete.

    Args:
        circles: Parameters of the circles for which to compute the circumferential completeness indices. Each circle
            must be defined by three parameters in the following order: x-coordinate of the center, y-coordinate of the
            center, radius.
        xy: Coordinates of the set of 2D points to which the circles were fitted.
        num_regions: Number of angular regions.
        max_dist: Maximum distance a point can have to the circle outline to be counted as part of the circle. If set to
            :code:`None`, points are counted as part of the circle if their distance to the circle is center is in the
            interval :math:`[0.7 \cdot r, 1.3 \cdot r]` where :math:`r` is the circle radius. Defaults to :code:`None`.

    Returns:
        Circumferential completeness indices of the circles.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`xy`: :math:`(N, 2)`
        - Output: :math:`(C)`

        | where
        |
        | :math:`C = \text{ number of circles}`
        | :math:`N = \text{ number of points}`
    """
    circumferential_completeness_indices = np.full(len(circles), fill_value=0, dtype=np.float64)

    angular_step_size = 2 * np.pi / num_regions

    circle: npt.NDArray[np.float64]
    for idx, circle in enumerate(circles):  # type: ignore[assignment]
        centered_points = xy - circle[:2]
        radii = np.linalg.norm(centered_points, axis=-1)

        if max_dist is None:
            circle_points = centered_points[np.logical_and(radii >= 0.7 * circle[2], radii <= 1.3 * circle[2])]
        else:
            circle_points = centered_points[np.abs(radii - circle[2]) <= max_dist]

        angles = np.arctan2(circle_points[:, 1], circle_points[:, 0])

        sections = np.remainder(np.floor(angles / angular_step_size).astype(np.int64), num_regions)
        filled_sections = np.unique(sections)

        circumferential_completeness_indices[idx] = len(filled_sections) / num_regions

    return circumferential_completeness_indices


def filter_circumferential_completeness_index(
    circles: npt.NDArray[np.float64],
    xy: npt.NDArray[np.float64],
    min_circumferential_completeness_index: float,
    num_regions: int,
    max_dist: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    r"""
    Filters out the circles whose circumferential completeness index is below the specified minimum circumferential
    completeness index.

    Args:
        circles: Parameters of the circles for which to compute the circumferential completeness indices. Each circle
            must be defined by three parameters in the following order: x-coordinate of the center, y-coordinate of the
            center, radius.
        xy: Coordinates of the set of 2D points to which the circles were fitted.
        num_regions: Number of angular regions.
        max_dist: Maximum distance a point can have to the circle outline to be counted as part of the circle. If set to
            :code:`None`, points are counted as part of the circle if their distance to the circle is center is in the
            interval :math:`[0.7 \cdot r, 1.3 \cdot r]` where :math:`r` is the circle radius. Defaults to :code:`None`.
        min_circumferential_completeness_index: Minimum circumferential index a point must have to not be discarded.

    Returns:
        Tuple consisting of two arrays. The first contains the parameters of the circles remaining after filtering. The
        second contains the indices of the retained circles in the original circle array.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`xy`: :math:`(N, 2)`
        - Output: Tuple of two arrays. The first has shape :math:`(C', 3)` and the second has shape :math:`(C)`.

        | where
        |
        | :math:`C = \text{ number of circles before the filtering}`
        | :math:`C' = \text{ number of circles after the filtering}`
        | :math:`N = \text{ number of points}`
    """

    circumferential_completeness_indices = circumferential_completeness_index(
        circles, xy, max_dist=max_dist, num_regions=num_regions
    )
    filter_mask = circumferential_completeness_indices >= min_circumferential_completeness_index

    selected_indices = np.arange(len(circles), dtype=np.int64)[filter_mask]
    circles = circles[selected_indices]

    return circles, selected_indices
