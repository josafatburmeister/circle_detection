"""Tests for :code:`circle_detection.operations.circumferential_completeness_index`."""

import multiprocessing
import time
from typing import Optional

import numpy as np
import pytest

from circle_detection.operations import circumferential_completeness_index, filter_circumferential_completeness_index

import sys

sys.path.insert(0, '..')

from test.utils import generate_circles, generate_circle_points  # pylint: disable=wrong-import-order


class TestCircumferentialCompletenessIndex:  # pylint: disable=too-few-public-methods
    """Tests for :code:`circle_detection.operations.circumferential_completeness_index`."""

    # @pytest.mark.parametrize("pass_batch_lengths", [True, False])
    # @pytest.mark.parametrize("max_dist", [0.1, None])
    # @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64])
    def test_circumferential_completeness_index(
        self, pass_batch_lengths: bool, max_dist: Optional[float], scalar_dtype: np.dtype
    ):
        circles = np.array([[0, 0, 1], [5, 0, 1]], dtype=scalar_dtype)
        xy = np.array(
            [
                [np.sqrt(0.5), np.sqrt(0.5)],
                [np.sqrt(0.5), -np.sqrt(0.5)],
                [-np.sqrt(0.5), np.sqrt(0.5)],
                [-np.sqrt(0.5), -np.sqrt(0.5)],
                [5, 1],
                [5, -1],
            ],
            dtype=scalar_dtype,
        )

        if pass_batch_lengths:
            batch_lengths_circles = np.array([len(circles)], dtype=np.int64)
            batch_lengths_xy = np.array([len(xy)], dtype=np.int64)
        else:
            batch_lengths_circles = None
            batch_lengths_xy = None

        num_regions = 4

        expected_circumferential_completness_indices = np.array([1, 0.5], dtype=scalar_dtype)

        circumferential_completeness_indices = circumferential_completeness_index(
            circles, xy, num_regions, max_dist, batch_lengths_circles, batch_lengths_xy
        )

        assert circumferential_completeness_indices.dtype == scalar_dtype
        np.testing.assert_array_equal(
            expected_circumferential_completness_indices, circumferential_completeness_indices
        )

        expected_filtered_circles = circles[:1]
        expected_batch_lengths_circles = np.array([1], dtype=np.int64)

        filtered_circles, batch_lengths_circles, selected_indices = filter_circumferential_completeness_index(
            circles,
            xy,
            min_circumferential_completeness_index=0.6,
            num_regions=num_regions,
            max_dist=max_dist,
            batch_lengths_circles=batch_lengths_circles,
            batch_lengths_xy=batch_lengths_xy,
        )

        assert filtered_circles.dtype == scalar_dtype
        np.testing.assert_array_equal(expected_filtered_circles, filtered_circles)
        np.testing.assert_array_equal(expected_batch_lengths_circles, batch_lengths_circles)
        np.testing.assert_array_equal(filtered_circles, circles[selected_indices])

    def test_batch_processing(self):
        circles = np.array([[0, 0, 1], [5, 0, 1], [7, 0, 1], [0, 0, 0.1]], dtype=np.float64)
        xy = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [5, 1], [5, -1], [7, 1], [7, -1], [8, 0]], dtype=np.float64)
        batch_lengths_circles = np.array([1, 2, 1], dtype=np.int64)
        batch_lengths_xy = np.array([4, 5, 0], dtype=np.int64)

        max_dist = 0.1
        num_regions = 4

        expected_circumferential_completness_indices = np.array([1, 0.5, 3 / 4, 0], dtype=np.float64)

        circumferential_completeness_indices = circumferential_completeness_index(
            circles, xy, num_regions, max_dist, batch_lengths_circles, batch_lengths_xy
        )

        np.testing.assert_array_equal(
            expected_circumferential_completness_indices, circumferential_completeness_indices
        )

        expected_filtered_circles = np.array([circles[0], circles[2]], dtype=np.float64)
        expected_filtered_batch_lengths_circles = np.array([1, 1, 0], dtype=np.int64)

        filtered_circles, filtered_batch_lengths_circles, selected_indices = filter_circumferential_completeness_index(
            circles,
            xy,
            min_circumferential_completeness_index=0.6,
            num_regions=num_regions,
            max_dist=max_dist,
            batch_lengths_circles=batch_lengths_circles,
            batch_lengths_xy=batch_lengths_xy,
        )

        np.testing.assert_array_equal(expected_filtered_circles, filtered_circles)
        np.testing.assert_array_equal(expected_filtered_batch_lengths_circles, filtered_batch_lengths_circles)
        np.testing.assert_array_equal(filtered_circles, circles[selected_indices])

    @pytest.mark.skipif(multiprocessing.cpu_count() <= 1, reason="Testing of multi-threading requires multiple cores.")
    def test_multi_threading(self):
        print("multiprocessing.cpu_count()", multiprocessing.cpu_count())

        batch_size = 1000
        circles = np.array([[[0, 0, 1], [5, 0, 1]]], dtype=np.float64)
        circles = generate_circles(
            num_circles=4,
            min_radius=0.2,
            max_radius=0.6,
        )
        xy = generate_circle_points(circles, min_points=2000, max_points=2000, add_noise_points=True, variance=0.01)

        batch_lengths_xy = np.array([len(xy)] * batch_size, dtype=np.int64)
        batch_lengths_circles = np.array([len(circles)] * batch_size, dtype=np.int64)
        circles = np.repeat(circles, batch_size, axis=0).reshape(-1, 3).copy(order="F")
        xy = np.repeat(xy, batch_size, axis=0).reshape(-1, 2).copy(order="F")

        max_dist = 0.1
        num_regions = 4

        single_threaded_runtime = 0
        multi_threaded_runtime = 0

        repetitions = 8
        for _ in range(repetitions):
            start = time.perf_counter()
            circumferential_completeness_index(
                circles, xy, num_regions, max_dist, batch_lengths_circles, batch_lengths_xy, num_workers=1
            )
            single_threaded_runtime += time.perf_counter() - start
            start = time.perf_counter()
            circumferential_completeness_index(
                circles, xy, num_regions, max_dist, batch_lengths_circles, batch_lengths_xy, num_workers=-1
            )
            multi_threaded_runtime += time.perf_counter() - start

        assert multi_threaded_runtime < single_threaded_runtime

    def test_invalid_batch_lengths_circles(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        batch_lengths_circles = np.array([2], dtype=np.int64)
        xy = np.zeros((10, 2), dtype=np.float64)
        batch_lengths_xy = np.array([len(xy)], dtype=np.int64)

        num_regions = 4

        with pytest.raises(ValueError):
            circumferential_completeness_index(
                circles, xy, num_regions, batch_lengths_circles=batch_lengths_circles, batch_lengths_xy=batch_lengths_xy
            )

        with pytest.raises(ValueError):
            filter_circumferential_completeness_index(
                circles,
                xy,
                num_regions,
                min_circumferential_completeness_index=0.6,
                batch_lengths_circles=batch_lengths_circles,
                batch_lengths_xy=batch_lengths_xy,
            )

    def test_invalid_batch_lengths_xy(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        batch_lengths_circles = np.array([len(circles)], dtype=np.int64)
        xy = np.zeros((10, 2), dtype=np.float64)
        batch_lengths_xy = np.array([5], dtype=np.int64)

        num_regions = 4

        with pytest.raises(ValueError):
            circumferential_completeness_index(
                circles, xy, num_regions, batch_lengths_circles=batch_lengths_circles, batch_lengths_xy=batch_lengths_xy
            )

        with pytest.raises(ValueError):
            filter_circumferential_completeness_index(
                circles,
                xy,
                num_regions,
                min_circumferential_completeness_index=0.6,
                batch_lengths_circles=batch_lengths_circles,
                batch_lengths_xy=batch_lengths_xy,
            )

    @pytest.mark.parametrize("omit_batch_lengths_circles", [True, False])
    def test_inconsistent_batch_lengths(self, omit_batch_lengths_circles: bool):
        circles = np.zeros((4, 3), dtype=np.float64)
        if omit_batch_lengths_circles:
            batch_lengths_circles = None
        else:
            batch_lengths_circles = np.array([2, 2], dtype=np.int64)
        xy = np.zeros((10, 2), dtype=np.float64)
        if not omit_batch_lengths_circles:
            batch_lengths_xy = None
        else:
            batch_lengths_xy = np.array([5, 5], dtype=np.int64)

        num_regions = 4

        with pytest.raises(ValueError):
            circumferential_completeness_index(
                circles, xy, num_regions, batch_lengths_circles=batch_lengths_circles, batch_lengths_xy=batch_lengths_xy
            )

        with pytest.raises(ValueError):
            filter_circumferential_completeness_index(
                circles,
                xy,
                num_regions,
                min_circumferential_completeness_index=0.6,
                batch_lengths_circles=batch_lengths_circles,
                batch_lengths_xy=batch_lengths_xy,
            )


if __name__ == "__main__":
    print("------")
    import platform, struct, sys
    print(platform.platform())
    print(platform.machine())
    print(struct.calcsize("P")*8, "bit")
    print(sys.executable)
    print("+++++++++++++")

    tests = TestCircumferentialCompletenessIndex()
    tests.test_circumferential_completeness_index(pass_batch_lengths=True, max_dist=None, scalar_dtype=np.float64)
    print("Passed")