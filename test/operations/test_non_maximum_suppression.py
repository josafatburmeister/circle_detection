""" Tests for :code:`circle_detection.operations.non_maximum_suppression`. """

import numpy as np
import pytest

from circle_detection.operations import non_maximum_suppression


class TestNonMaximumSuppression:
    """Tests for :code:`circle_detection.operations.non_maximum_suppression`."""

    @pytest.mark.parametrize("pass_batch_lengths", [True, False])
    def test_non_overlapping_circles(self, pass_batch_lengths: bool):
        circles = np.array([[0, 0, 1], [3, 0, 0.5], [0, 2, 0.1]], dtype=np.float64)
        fitting_losses = np.zeros(3, dtype=np.float64)
        batch_lengths = np.array([3], dtype=np.int64)

        filtered_circles, filtered_fitting_losses, filtered_batch_lengths, selected_indices = non_maximum_suppression(
            circles, fitting_losses, batch_lengths if pass_batch_lengths else None
        )

        np.testing.assert_array_equal(circles, filtered_circles)
        np.testing.assert_array_equal(fitting_losses, filtered_fitting_losses)
        np.testing.assert_array_equal(batch_lengths, filtered_batch_lengths)
        np.testing.assert_array_equal(filtered_circles, circles[selected_indices])

    def test_overlapping_circles(self):
        circles = np.array([[0, 0, 1], [0.9, 0.1, 1], [0, 2, 0.1]], dtype=np.float64)
        fitting_losses = np.array([-1, -2, -3], dtype=np.float64)
        batch_lengths = np.array([len(circles)], dtype=np.int64)

        expected_filtered_circles = np.array([[0, 2, 0.1], [0.9, 0.1, 1]], dtype=np.float64)
        expected_filtered_fitting_losses = np.array([-3, -2], dtype=np.float64)
        expected_filtered_batch_lengths = np.array([len(expected_filtered_circles)], dtype=np.int64)

        filtered_circles, filtered_fitting_losses, filtered_batch_lengths, selected_indices = non_maximum_suppression(
            circles, fitting_losses, batch_lengths
        )

        np.testing.assert_array_equal(expected_filtered_circles, filtered_circles)
        np.testing.assert_array_equal(expected_filtered_fitting_losses, filtered_fitting_losses)
        np.testing.assert_array_equal(expected_filtered_batch_lengths, filtered_batch_lengths)
        np.testing.assert_array_equal(filtered_circles, circles[selected_indices])

    def test_batch_processing(self):
        circles = np.array([[0, 0, 1], [0, 0, 0.9], [0, 0, 0.8], [0, 0, 0.7], [0, 0, 0.6]], dtype=np.float64)
        fitting_losses = np.array([-5, -4, -3, -2, -1], dtype=np.float64)
        batch_lengths = np.array([2, 3], dtype=np.int64)

        expected_filtered_circles = np.array([[0, 0, 1], [0, 0, 0.8]], dtype=np.float64)
        expected_filtered_fitting_losses = np.array([-5, -3], dtype=np.float64)
        expected_filtered_batch_lengths = np.array([1, 1], dtype=np.int64)

        filtered_circles, filtered_fitting_losses, filtered_batch_lengths, selected_indices = non_maximum_suppression(
            circles, fitting_losses, batch_lengths
        )

        np.testing.assert_array_equal(expected_filtered_circles, filtered_circles)
        np.testing.assert_array_equal(expected_filtered_fitting_losses, filtered_fitting_losses)
        np.testing.assert_array_equal(expected_filtered_batch_lengths, filtered_batch_lengths)
        np.testing.assert_array_equal(filtered_circles, circles[selected_indices])

    def test_invalid_inputs(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        fitting_losses = np.zeros((2), dtype=np.float64)

        with pytest.raises(ValueError):
            non_maximum_suppression(circles, fitting_losses)

    def test_invalid_batch_lengths(self):
        circles = np.zeros((4, 3), dtype=np.float64)
        fitting_losses = np.zeros((4), dtype=np.float64)
        batch_lengths = np.array([2], dtype=np.int64)

        with pytest.raises(ValueError):
            non_maximum_suppression(circles, fitting_losses, batch_lengths)
