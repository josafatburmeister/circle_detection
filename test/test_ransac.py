""" Tests for circle_detection.Ransac. """

import multiprocessing
import time
from typing import Any, Dict, Optional

import numpy as np
import pytest

from circle_detection import Ransac

from test.utils import generate_circle_points  # pylint: disable=wrong-import-order


class TestRansac:
    """Tests for circle_detection.Ransac."""

    @pytest.mark.parametrize("add_noise_points", [True, False])
    @pytest.mark.parametrize("seed", [1, None])
    @pytest.mark.parametrize("storage_layout", ["C", "F"])
    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64])
    def test_circle_fitting(  # pylint: disable=too-many-locals
        self, add_noise_points: bool, seed: Optional[int], storage_layout: str, scalar_dtype: np.dtype
    ):
        batch_size = 250
        circles = []
        xy = []
        batch_lengths = []
        for batch_idx in range(batch_size):
            random_generator = np.random.default_rng(batch_idx)
            center_x = random_generator.uniform(0, batch_size, 1)[0]
            center_y = random_generator.uniform(0, batch_size, 1)[0]
            radius = random_generator.uniform(0, 1, 1)[0]
            current_circles = np.array([[center_x, center_y, radius]], dtype=scalar_dtype)
            circles.append(current_circles)
            current_xy = generate_circle_points(
                current_circles, min_points=50, max_points=500, add_noise_points=add_noise_points, seed=batch_idx
            )
            xy.append(current_xy)
            batch_lengths.append(len(current_xy))

        ransac = Ransac(bandwidth=0.01, iterations=500)
        ransac.detect(
            np.concatenate(xy).astype(scalar_dtype).copy(order=storage_layout),
            batch_lengths=np.array(batch_lengths, dtype=np.int64),
            num_workers=-1,
            seed=seed,
        )
        ransac.filter(max_circles=1, deduplication_precision=4, non_maximum_suppression=False)

        expected_circles = np.concatenate(circles).astype(scalar_dtype)

        assert len(expected_circles) == len(ransac.circles)
        assert ransac.circles.dtype == scalar_dtype
        assert ransac.fitting_scores.dtype == scalar_dtype

        if add_noise_points:
            invalid_mask = np.abs((ransac.circles - expected_circles)).sum(axis=-1) > 1e-3
            assert invalid_mask.sum() < len(expected_circles) * 0.02
        else:
            np.testing.assert_almost_equal(expected_circles, ransac.circles, decimal=4)

    @pytest.mark.skipif(multiprocessing.cpu_count() <= 1, reason="Testing of multi-threading requires multiple cores.")
    def test_multi_threading(self):
        original_circles = np.array([[0, 0, 0.5], [0, 0, 0.52]])
        xy_1 = generate_circle_points(original_circles[:1], min_points=500, max_points=500, variance=0.0)
        xy_2 = generate_circle_points(original_circles[1:], min_points=500, max_points=500, variance=0.0)
        batch_lengths = np.array([len(xy_1), len(xy_2)], dtype=np.int64)

        circle_detector = Ransac(bandwidth=0.05, iterations=2000)

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

        circle_detector = Ransac(bandwidth=bandwidth)
        circle_detector.detect(xy, num_workers=-1)
        circle_detector.filter(max_circles=1, num_workers=-1)

        # check that return values are passed by reference
        assert circle_detector.circles.flags.owndata is False
        assert circle_detector.fitting_scores.flags.owndata is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_samples": 2},
            {"min_concensus_points": 2},
        ],
    )
    def test_invalid_constructor_parameters(self, kwargs: Dict[str, Any]):
        args: Dict[str, Any] = {"bandwidth": 0.01}

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        with pytest.raises(ValueError):
            Ransac(**args)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"break_min_x": 1, "break_max_x": 0},
            {"break_min_y": 1, "break_max_y": 0},
            {"break_min_radius": 1, "break_max_radius": 0.1},
            {"batch_lengths": np.array([], dtype=np.int64)},
            {"batch_lengths": np.array([99], dtype=np.int64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_min_x": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_x": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_min_y": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_y": np.ones(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_min_radius": np.zeros(3, dtype=np.float64)},
            {"batch_lengths": np.array([50, 50], dtype=np.int64), "break_max_radius": np.ones(3, dtype=np.float64)},
        ],
    )
    def test_invalid_detection_parameters(self, kwargs: Dict[str, Any]):
        xy = np.zeros((100, 2), dtype=np.float64)

        args: Dict[str, Any] = {
            "break_min_x": -1,
            "break_max_x": 1,
            "break_min_y": -1,
            "break_max_y": 1,
            "break_min_radius": 0,
            "break_max_radius": 2,
        }

        for arg_name, arg_value in kwargs.items():
            args[arg_name] = arg_value

        circle_detector = Ransac(bandwidth=0.01)

        with pytest.raises(ValueError):
            circle_detector.detect(
                xy,
                **args,
            )
