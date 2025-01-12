""" Tests for :code:`circle_detection.operations.select_top_k_circles`. """

import numpy as np
import pytest

from circle_detection.operations import select_top_k_circles


class TestSelectTopKCircles:
    """Tests for :code:`circle_detection.operations.select_top_k_circles`."""

    @pytest.mark.parametrize("pass_batch_lengths", [True, False])
    @pytest.mark.parametrize("scalar_dtype", [np.float32, np.float64])
    def test_single_batch_item(self, pass_batch_lengths: bool, scalar_dtype: np.dtype):
        circles = np.array([[0, 0, 1], [0, 0, 0.9], [0, 0, 0.8]], dtype=scalar_dtype)
        fitting_scores = np.array([1, 3, 2], dtype=scalar_dtype)
        batch_lengths = None
        if pass_batch_lengths:
            batch_lengths = np.array([3], dtype=np.int64)

        expected_circles = np.array([[0, 0, 0.9], [0, 0, 0.8]], dtype=scalar_dtype)
        expected_fitting_scores = np.array([3, 2], dtype=scalar_dtype)
        expected_batch_lengths = np.array([2])
        expected_selected_indices = np.array([1, 2])

        k = 2
        selected_circles, selected_fitting_scores, selected_batch_lengths, selected_indices = select_top_k_circles(
            circles, fitting_scores, k=k, batch_lengths=batch_lengths
        )

        assert selected_circles.dtype == scalar_dtype
        assert selected_fitting_scores.dtype == scalar_dtype

        np.testing.assert_array_equal(expected_circles, selected_circles)
        np.testing.assert_array_equal(expected_fitting_scores, selected_fitting_scores)
        np.testing.assert_array_equal(expected_batch_lengths, selected_batch_lengths)
        np.testing.assert_array_equal(expected_selected_indices, selected_indices)

    def test_batch_processing(self):
        circles = np.array(
            [[0, 0, 1], [0, 0, 0.9], [0, 0, 0.8], [0, 0, 0.7], [0, 0, 0.6], [0, 0, 0.5]], dtype=np.float64
        )
        fitting_scores = np.array([1, 3, 2, 1, 2, 1], dtype=np.float64)
        batch_lengths = np.array([3, 2, 1], dtype=np.int64)

        expected_circles = np.array([[0, 0, 0.9], [0, 0, 0.8], [0, 0, 0.7], [0, 0, 0.6], [0, 0, 0.5]], dtype=np.float64)
        expected_fitting_scores = np.array([3, 2, 1, 2, 1], dtype=np.float64)
        expected_batch_lengths = np.array([2, 2, 1])
        expected_selected_indices = np.array([1, 2, 3, 4, 5])

        k = 2
        selected_circles, selected_fitting_scores, selected_batch_lengths, selected_indices = select_top_k_circles(
            circles, fitting_scores, k=k, batch_lengths=batch_lengths
        )

        np.testing.assert_array_equal(expected_circles, selected_circles)
        np.testing.assert_array_equal(expected_fitting_scores, selected_fitting_scores)
        np.testing.assert_array_equal(expected_batch_lengths, selected_batch_lengths)
        np.testing.assert_array_equal(expected_selected_indices, selected_indices)

    def test_k_larger_than_input(self):
        circles = np.array([[0, 0, 0.5]], dtype=np.float64)
        fitting_scores = np.array([0], dtype=np.float64)
        k = 2
        selected_circles, selected_fitting_scores, selected_batch_lengths, selected_indices = select_top_k_circles(
            circles, fitting_scores, k=k
        )

        np.testing.assert_array_equal(circles, selected_circles)
        np.testing.assert_array_equal(fitting_scores, selected_fitting_scores)
        np.testing.assert_array_equal(np.array([1], dtype=np.int64), selected_batch_lengths)
        np.testing.assert_array_equal(np.array([0], dtype=np.int64), selected_indices)

    def test_invalid_inputs(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        fitting_scores = np.zeros((2), dtype=np.float64)

        with pytest.raises(ValueError):
            select_top_k_circles(circles, fitting_scores, k=1)

    def test_invalid_batch_lengths(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        fitting_scores = np.zeros((4), dtype=np.float64)
        batch_lengths = np.array([2], dtype=np.int64)

        with pytest.raises(ValueError):
            select_top_k_circles(circles, fitting_scores, k=1, batch_lengths=batch_lengths)
