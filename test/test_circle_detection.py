""" Tests for :code:`circle_detection.CircleDetection`. """

import multiprocessing
import time
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import pytest

from circle_detection import CircleDetection


class TestCircleDetection:
    """Tests for :code:`circle_detection.CircleDetection`."""

    def _generate_circles(  # pylint: disable=too-many-locals
        self,
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

    def _generate_circle_points(  # pylint: disable=too-many-locals
        self,
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

    def test_circle_one_perfect_fit_and_one_noisy_circle(self):
        original_circles = np.array([[0, 0, 0.5], [0, 2, 0.5]])
        xy = self._generate_circle_points(
            original_circles, min_points=100, max_points=100, variance=np.array([0, 0.05])
        )
        bandwidth = 0.07

        circle_detection = CircleDetection(bandwidth=bandwidth)
        circle_detection.detect(xy, num_workers=-1)
        circle_detection.filter(max_circles=1, num_workers=-1)

        assert len(circle_detection.circles) == 1
        assert len(circle_detection.fitting_losses) == 1
        np.testing.assert_array_equal(circle_detection.batch_lengths_circles, np.array([1], dtype=np.int64))

        # the first circle is expected to be returned because its points have lower variance
        np.testing.assert_almost_equal(original_circles[0], circle_detection.circles[0], decimal=10)

        circle_detection.detect(xy, num_workers=-1)
        circle_detection.filter(max_circles=2, non_maximum_suppression=True, num_workers=-1)

        assert len(circle_detection.circles) == 2
        assert len(circle_detection.fitting_losses) == 2
        np.testing.assert_array_equal(circle_detection.batch_lengths_circles, np.array([2], dtype=np.int64))

        expected_fitting_losses = []
        for circle in circle_detection.circles:
            residuals = (np.linalg.norm(xy - circle[:2], axis=-1) - circle[2]) / bandwidth
            expected_loss = -1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * residuals**2)
            expected_fitting_losses.append(expected_loss.sum())

        assert (np.abs(original_circles - circle_detection.circles) < 0.01).all()
        np.testing.assert_almost_equal(expected_fitting_losses, circle_detection.fitting_losses, decimal=5)

    def test_several_noisy_circles(self):
        original_circles = self._generate_circles(
            num_circles=2,
            min_radius=0.2,
            max_radius=0.6,
        )
        xy = self._generate_circle_points(
            original_circles, min_points=50, max_points=150, add_noise_points=True, variance=0.01
        )

        min_start_xy = xy.min(axis=0) - 2
        max_start_xy = xy.max(axis=0) + 2

        circle_detection = CircleDetection(bandwidth=0.05, min_fitting_score=5)
        circle_detection.detect(
            xy,
            min_start_x=min_start_xy[0],
            max_start_x=max_start_xy[0],
            n_start_x=10,
            min_start_y=min_start_xy[1],
            max_start_y=max_start_xy[1],
            n_start_y=10,
            min_start_radius=0.1,
            max_start_radius=0.9,
            n_start_radius=10,
            break_min_x=min_start_xy[0],
            break_max_x=max_start_xy[0],
            break_min_y=min_start_xy[1],
            break_max_y=max_start_xy[1],
            break_min_radius=0,
            break_max_radius=1.5,
            num_workers=-1,
        )
        circle_detection.filter(deduplication_precision=3, non_maximum_suppression=True)

        assert len(original_circles) == len(circle_detection.circles)
        assert len(circle_detection.circles) == len(circle_detection.fitting_losses)
        np.testing.assert_array_equal(
            circle_detection.batch_lengths_circles, np.array([len(original_circles)], dtype=np.int64)
        )

        for original_circle in original_circles:
            matches_with_detected_circle = False
            for detected_circle in circle_detection.circles:
                if (np.abs(original_circle - detected_circle) < 0.03).all():
                    matches_with_detected_circle = True
                    break

            assert matches_with_detected_circle

    @pytest.mark.parametrize("pass_max_dist", [True, False])
    def test_filtering_circumferential_completeness_index(self, pass_max_dist: bool):
        original_circles = np.array([[0, 0, 0.5]])
        xy = self._generate_circle_points(original_circles, min_points=100, max_points=100, variance=0)
        bandwidth = 0.01

        max_dist = None
        if pass_max_dist:
            max_dist = bandwidth

        circle_detection = CircleDetection(bandwidth=bandwidth)
        circle_detection.detect(
            xy,
            num_workers=-1,
        )
        circle_detection.filter(
            max_circles=1,
            min_circumferential_completeness_idx=0.9,
            circumferential_completeness_idx_max_dist=max_dist,
            circumferential_completeness_idx_num_regions=int(365 / 5),
            num_workers=-1,
        )

        assert len(circle_detection.circles) == 1
        assert len(circle_detection.fitting_losses) == 1

        np.testing.assert_almost_equal(original_circles[0], circle_detection.circles[0], decimal=10)

        circle_detection.detect(
            xy[:50],
            num_workers=-1,
        )
        circle_detection.filter(
            max_circles=1,
            min_circumferential_completeness_idx=0.9,
            circumferential_completeness_idx_max_dist=max_dist,
            circumferential_completeness_idx_num_regions=int(365 / 5),
            num_workers=-1,
        )

        assert len(circle_detection.circles) == 0
        assert len(circle_detection.fitting_losses) == 0

    def test_batch_processing(self):
        original_circles = np.array([[0, 0, 0.5], [0, 0, 0.52]])
        xy_1 = self._generate_circle_points(original_circles[:1], min_points=100, max_points=100, variance=0.0)
        xy_2 = self._generate_circle_points(original_circles[1:], min_points=100, max_points=100, variance=0.0)
        batch_lengths = np.array([len(xy_1), len(xy_2)], dtype=np.int64)

        circle_detection = CircleDetection(bandwidth=0.05)

        circle_detection.detect(
            np.concatenate((xy_1, xy_2)),
            batch_lengths=batch_lengths,
            num_workers=-1,
        )
        circle_detection.filter(max_circles=1, num_workers=-1)

        num_batches = len(batch_lengths)
        assert len(circle_detection.circles) == num_batches
        assert len(circle_detection.fitting_losses) == num_batches

        batch_starts = np.cumsum(np.concatenate((np.array([0]), circle_detection.batch_lengths_circles)))[:-1]

        for batch_idx in range(num_batches):
            batch_start = batch_starts[batch_idx]
            batch_end = batch_start + circle_detection.batch_lengths_circles[batch_idx]
            np.testing.assert_almost_equal(
                original_circles[batch_idx].reshape(-1, 3), circle_detection.circles[batch_start:batch_end], decimal=5
            )

    @pytest.mark.skipif(multiprocessing.cpu_count() <= 1, reason="Testing of multi-threading requires multiple cores.")
    def test_multi_threading(self):
        original_circles = np.array([[0, 0, 0.5], [0, 0, 0.52]])
        xy_1 = self._generate_circle_points(original_circles[:1], min_points=100, max_points=100, variance=0.0)
        xy_2 = self._generate_circle_points(original_circles[1:], min_points=100, max_points=100, variance=0.0)
        batch_lengths = np.array([len(xy_1), len(xy_2)], dtype=np.int64)

        circle_detection = CircleDetection(bandwidth=0.05)

        single_threaded_runtime = 0
        multi_threaded_runtime = 0

        repetitions = 4
        for _ in range(repetitions):
            start = time.time()
            circle_detection.detect(
                np.concatenate((xy_1, xy_2)),
                batch_lengths=batch_lengths,
                num_workers=1,
            )
            single_threaded_runtime += time.time() - start
            start = time.time()
            circle_detection.detect(
                np.concatenate((xy_1, xy_2)),
                batch_lengths=batch_lengths,
                num_workers=-1,
            )
            multi_threaded_runtime += time.time() - start

        assert multi_threaded_runtime < single_threaded_runtime

    def test_empty_input(self):
        xy = np.empty((0, 2), dtype=np.float64)
        circle_detection = CircleDetection(bandwidth=0.05)
        circle_detection.detect(xy)

        assert len(circle_detection.circles) == 0
        assert len(circle_detection.fitting_losses) == 0
        assert circle_detection.batch_lengths_circles.sum() == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"acceleration_factor": 0.5},
            {"armijo_attenuation_factor": -1},
            {"armijo_attenuation_factor": 2},
            {"armijo_min_decrease_percentage": -1},
            {"armijo_min_decrease_percentage": 2},
        ],
    )
    def test_invalid_constructor_parameters(self, kwargs: Dict[str, Any]):
        args: Dict[str, Any] = {"bandwidth": 0.01}

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        with pytest.raises(ValueError):
            CircleDetection(**args)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_start_x": 1, "max_start_x": 0},
            {"min_start_x": 0, "break_min_x": 1},
            {"max_start_x": 1, "break_max_x": 0},
            {"min_start_y": 1, "max_start_y": 0},
            {"min_start_y": 0, "break_min_y": 1},
            {"max_start_y": 1, "break_max_y": 0},
            {"min_start_radius": 1, "max_start_radius": 0.1},
            {"min_start_radius": 0.1, "break_min_radius": 1},
            {"max_start_radius": 1, "break_max_radius": 0.1},
            {"min_start_radius": -1},
            {"n_start_y": -1},
            {"n_start_x": -1},
            {"n_start_radius": -1},
            {"batch_lengths": np.array([], dtype=np.int64)},
        ],
    )
    def test_invalid_detection_parameters(self, kwargs: Dict[str, Any]):
        xy = np.zeros((100, 2), dtype=np.float64)

        args: Dict[str, Any] = {
            "min_start_x": -1,
            "max_start_x": 1,
            "n_start_x": 1,
            "min_start_y": -1,
            "max_start_y": 1,
            "n_start_y": 1,
            "min_start_radius": 0.1,
            "max_start_radius": 1,
            "n_start_radius": 1,
            "break_min_x": -1,
            "break_max_x": 1,
            "break_min_y": -1,
            "break_max_y": 1,
            "break_min_radius": 0,
            "break_max_radius": 2,
        }

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        circle_detection = CircleDetection(bandwidth=0.01)

        with pytest.raises(ValueError):
            circle_detection.detect(
                xy,
                **args,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_circumferential_completeness_idx": 0.5, "circumferential_completeness_idx_num_regions": None},
        ],
    )
    def test_invalid_filtering_parameters(self, kwargs: Dict[str, Any]):
        xy = np.zeros((100, 2), dtype=np.float64)

        args: Dict[str, Any] = {}

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        circle_detection = CircleDetection(bandwidth=0.01)
        circle_detection.detect(xy)

        with pytest.raises(ValueError):
            circle_detection.filter(**args)

    def test_filtering_without_detection(self):
        circle_detection = CircleDetection(bandwidth=0.01)
        with pytest.raises(ValueError):
            circle_detection.filter()
