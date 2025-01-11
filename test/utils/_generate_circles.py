""" Utilities for generating test data. """

__all__ = ["generate_circles", "generate_circle_points"]

from typing import Union

import numpy as np
import numpy.typing as npt


def generate_circles(  # pylint: disable=too-many-locals
    num_circles: int,
    min_radius: float,
    max_radius: float,
    allow_overlapping_circles: bool = False,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """
    Randomly generates a set of 2D circles.

    Args:
        num_circles: Number of circles to generate.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        allow_overlapping_circles: Whether the generated circles are allowed to overlap. Defaults to :code:`False`.
        seed: Random seed. Defaults to 0.

    Returns:
        Parameters of the generated circles (in the following order: x-coordinate of the center, y-coordinate of the
        center, radius).

    Raises:
        ValueError: If :code:`min_radius` is larger than :code:`max_radius`.
    """

    random_generator = np.random.default_rng(seed=seed)

    circles = np.zeros((num_circles, 3))

    if min_radius < max_radius:
        circles[:, 2] = random_generator.uniform(min_radius, max_radius, num_circles)
    elif min_radius == max_radius:
        circles[:, 2] = np.full(num_circles, fill_value=min_radius, dtype=np.float64)
    else:
        raise ValueError("Minimum radius must not be larger than maximum radius.")

    min_xy = -2 * max_radius * num_circles
    max_xy = 2 * max_radius * num_circles

    for circle_idx in range(num_circles):
        draw_new_center = True

        while draw_new_center:
            radius = circles[circle_idx, 2]
            center = random_generator.uniform(min_xy, max_xy, 2)

            if not allow_overlapping_circles:
                circle_overlaps_with_previous_circles = False
                for previous_circle_idx in range(circle_idx):
                    dist = np.linalg.norm(circles[previous_circle_idx, :2] - center)  # type: ignore[attr-defined]
                    if dist <= circles[previous_circle_idx, 2] + radius:
                        circle_overlaps_with_previous_circles = True
                        break
                if not circle_overlaps_with_previous_circles:
                    break
            else:
                break

        circles[circle_idx, :2] = center

    return circles


def generate_circle_points(  # pylint: disable=too-many-locals
    circles: npt.NDArray[np.float64],
    min_points: int,
    max_points: int,
    add_noise_points: bool = False,
    seed: int = 0,
    variance: Union[float, npt.NDArray[np.float64]] = 0,
) -> npt.NDArray[np.float64]:
    """
    Generates a set of 2D points that are randomly sampled around the outlines of the specified circles.

    Args:
        circles: Parameters of the circles from which to sample (in the following order: x-coordinate of the center,
            y-coordinate of the center, radius).
        min_points: Minimum number of points to sample from each circle.
        max_points: Maximum number of points to sample from each circle.
        add_noise_points: Whether randomly placed noise points not sampled from a circle should be added to the set
            of 2D points. Defaults to :code:`False`.
        seed: Random seed. Defaults to 0.
        variance: Variance of the distance of the sampled points to the circle outlines. Can be either a scalar
            value or an array of values whose length is equal to :code:`num_circles`.

    Returns:
        Tuple of two arrays. The first contains the parameters of the generated circles (in the order x, y and
        radius). The second contains the x- and y-coordinates of the generated 2D points.

    Raises:
        ValueError: If :code:`variance` is an arrays whose length is not equal to :code:`circles`.
    """
    xy = []
    random_generator = np.random.default_rng(seed=seed)

    if isinstance(variance, np.ndarray) and len(variance) != len(circles):
        raise ValueError("Length of variance must be equal to num_circles.")

    circle: npt.NDArray[np.float64]
    for circle_idx, circle in enumerate(circles):  # type: ignore[assignment]
        num_points = int(random_generator.uniform(min_points, max_points))

        angles = np.linspace(0, 2 * np.pi, num_points)
        point_radii = np.full(num_points, fill_value=circle[2], dtype=np.float64)

        if isinstance(variance, (float, int)):
            current_variance = float(variance)
        else:
            current_variance = variance[circle_idx]

        point_radii += random_generator.normal(0, current_variance, num_points)

        x = point_radii * np.cos(angles)
        y = point_radii * np.sin(angles)
        xy.append(np.column_stack([x, y]) + circle[:2])

    if add_noise_points:
        num_points = int(random_generator.uniform(min_points * 0.1, max_points * 0.1))
        min_xy = (circles[:, :2] - circles[:, 2]).min(axis=0).min()
        max_xy = (circles[:, :2] + circles[:, 2]).max(axis=0).max()
        noise_points = random_generator.uniform(min_xy, max_xy, (num_points, 2))
        xy.append(noise_points)

    return np.concatenate(xy)
