"""Tests for :code:`circle_detection.MEstimator`."""

import multiprocessing
import time
from typing import Any, Dict

import numpy as np
import pytest

from circle_detection import MEstimator

from test.utils import generate_circles, generate_circle_points  # pylint: disable=wrong-import-order


class TestMEstimator:
    """Tests for :code:`circle_detection.MEstimator`."""

    @pytest.mark.parametrize("storage_layout", ["C", "F"])
    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64])
    def test_circle_one_perfect_fit_and_one_noisy_circle(self, storage_layout: str, scalar_dtype: np.dtype):
        original_circles = np.array([[0, 0, 0.5], [0, 2, 0.5]])
        xy = generate_circle_points(original_circles, min_points=100, max_points=100, variance=np.array([0, 0.05]))
        xy = xy.copy(order=storage_layout)
        xy = xy.astype(scalar_dtype)
        bandwidth = 0.07

        circle_detector = MEstimator(bandwidth=bandwidth)
        circle_detector.detect(xy, num_workers=-1)
        circle_detector.filter(max_circles=1, num_workers=-1)

        assert len(circle_detector.circles) == 1
        assert len(circle_detector.fitting_scores) == 1
        assert circle_detector.circles.dtype == scalar_dtype
        assert circle_detector.fitting_scores.dtype == scalar_dtype
        np.testing.assert_array_equal(circle_detector.batch_lengths_circles, np.array([1], dtype=np.int64))

        # the first circle is expected to be returned because its points have lower variance
        decimal = 10 if scalar_dtype == np.float64 else 5
        np.testing.assert_almost_equal(original_circles[0], circle_detector.circles[0], decimal=decimal)

        circle_detector.detect(xy, num_workers=-1)
        circle_detector.filter(max_circles=2, non_maximum_suppression=True, num_workers=-1)

        assert len(circle_detector.circles) == 2
        assert len(circle_detector.fitting_scores) == 2
        assert circle_detector.circles.dtype == scalar_dtype
        assert circle_detector.fitting_scores.dtype == scalar_dtype
        np.testing.assert_array_equal(circle_detector.batch_lengths_circles, np.array([2], dtype=np.int64))

        expected_fitting_scores = []
        for circle in circle_detector.circles:
            residuals = (np.linalg.norm(xy - circle[:2], axis=-1) - circle[2]) / bandwidth
            expected_fitting_score = 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * residuals**2) / bandwidth
            expected_fitting_scores.append(expected_fitting_score.sum())

        assert (np.abs(original_circles - circle_detector.circles) < 0.01).all()
        decimal = 10 if scalar_dtype == np.float64 else 4
        np.testing.assert_almost_equal(expected_fitting_scores, circle_detector.fitting_scores, decimal=decimal)

    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64])
    def test_several_noisy_circles(self, scalar_dtype: np.dtype):
        original_circles = generate_circles(
            num_circles=2,
            min_radius=0.2,
            max_radius=0.6,
        )
        xy = generate_circle_points(
            original_circles, min_points=50, max_points=150, add_noise_points=True, variance=0.01
        )
        xy = xy.astype(scalar_dtype)

        min_start_xy = xy.min(axis=0) - 2
        max_start_xy = xy.max(axis=0) + 2

        circle_detector = MEstimator(bandwidth=0.05, min_fitting_score=100)
        circle_detector.detect(
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
        circle_detector.filter(deduplication_precision=3, non_maximum_suppression=True)

        assert len(original_circles) == len(circle_detector.circles)
        assert len(circle_detector.circles) == len(circle_detector.fitting_scores)
        assert circle_detector.circles.dtype == scalar_dtype
        assert circle_detector.fitting_scores.dtype == scalar_dtype
        np.testing.assert_array_equal(
            circle_detector.batch_lengths_circles, np.array([len(original_circles)], dtype=np.int64)
        )

        for original_circle in original_circles:
            matches_with_detected_circle = False
            for detected_circle in circle_detector.circles:
                if (np.abs(original_circle - detected_circle) < 0.03).all():
                    matches_with_detected_circle = True
                    break

            assert matches_with_detected_circle

    def test_batch_processing(self):
        original_circles = np.array([[0, 0, 0.5], [0, 0, 0.52]])
        xy_1 = generate_circle_points(original_circles[:1], min_points=100, max_points=100, variance=0.0)
        xy_2 = generate_circle_points(original_circles[1:], min_points=100, max_points=100, variance=0.0)
        batch_lengths = np.array([len(xy_1), len(xy_2)], dtype=np.int64)

        circle_detector = MEstimator(bandwidth=0.05)

        circle_detector.detect(
            np.concatenate((xy_1, xy_2)),
            batch_lengths=batch_lengths,
            num_workers=-1,
        )
        circle_detector.filter(max_circles=1, num_workers=-1)

        num_batches = len(batch_lengths)
        assert len(circle_detector.circles) == num_batches
        assert len(circle_detector.fitting_scores) == num_batches

        batch_starts = np.cumsum(np.concatenate((np.array([0]), circle_detector.batch_lengths_circles)))[:-1]

        for batch_idx in range(num_batches):
            batch_start = batch_starts[batch_idx]
            batch_end = batch_start + circle_detector.batch_lengths_circles[batch_idx]
            np.testing.assert_almost_equal(
                original_circles[batch_idx].reshape(-1, 3), circle_detector.circles[batch_start:batch_end], decimal=5
            )

    @pytest.mark.skipif(multiprocessing.cpu_count() <= 1, reason="Testing of multi-threading requires multiple cores.")
    def test_multi_threading(self):
        original_circles = np.array([[0, 0, 0.5], [0, 0, 0.52]])
        xy_1 = generate_circle_points(original_circles[:1], min_points=100, max_points=100, variance=0.0)
        xy_2 = generate_circle_points(original_circles[1:], min_points=100, max_points=100, variance=0.0)
        batch_lengths = np.array([len(xy_1), len(xy_2)], dtype=np.int64)

        circle_detector = MEstimator(bandwidth=0.05)

        single_threaded_runtime = 0
        multi_threaded_runtime = 0

        repetitions = 4
        for _ in range(repetitions):
            start = time.perf_counter()
            circle_detector.detect(
                np.concatenate((xy_1, xy_2)),
                batch_lengths=batch_lengths,
                num_workers=1,
            )
            single_threaded_runtime += time.perf_counter() - start
            start = time.perf_counter()
            circle_detector.detect(
                np.concatenate((xy_1, xy_2)),
                batch_lengths=batch_lengths,
                num_workers=-1,
            )
            multi_threaded_runtime += time.perf_counter() - start

        assert multi_threaded_runtime < single_threaded_runtime

    def test_return_values_pass_by_reference(self):
        original_circles = np.array([[0, 0, 0.5]])
        xy = generate_circle_points(original_circles, min_points=100, max_points=100)
        bandwidth = 0.05

        circle_detector = MEstimator(bandwidth=bandwidth)
        circle_detector.detect(xy, num_workers=-1)
        circle_detector.filter(max_circles=1, num_workers=-1)

        # check that return values are passed by reference
        assert circle_detector.circles.flags.owndata is False
        assert circle_detector.fitting_scores.flags.owndata is False

    def test_empty_input(self):
        xy = np.empty((0, 2), dtype=np.float64)
        circle_detector = MEstimator(bandwidth=0.05)
        circle_detector.detect(xy)

        assert len(circle_detector.circles) == 0
        assert len(circle_detector.fitting_scores) == 0
        assert circle_detector.batch_lengths_circles.sum() == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"acceleration_factor": 0.5},
            {"armijo_attenuation_factor": -1},
            {"armijo_attenuation_factor": 2},
            {"armijo_min_decrease_percentage": -1},
            {"armijo_min_decrease_percentage": 2},
            {"min_step_size": 0.01, "break_min_change": 0.001},
        ],
    )
    def test_invalid_constructor_parameters(self, kwargs: Dict[str, Any]):
        args: Dict[str, Any] = {"bandwidth": 0.01}

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        with pytest.raises(ValueError):
            MEstimator(**args)

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
            {"batch_lengths": np.array([99], dtype=np.int64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "min_start_x": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "max_start_x": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_x": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "min_start_y": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "max_start_y": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_min_y": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_y": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "min_start_radius": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "max_start_radius": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_min_radius": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_radius": np.ones(3, dtype=np.float64)},
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

        circle_detector = MEstimator(bandwidth=0.01)

        with pytest.raises(ValueError):
            circle_detector.detect(
                xy,
                **args,
            )
