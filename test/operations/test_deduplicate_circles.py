""" Tests for circle_detection.operations.deduplicate_circles. """

import numpy as np
import pytest

from circle_detection.operations import deduplicate_circles


class TestDeduplicateCircles:
    """Tests for circle_detection.operations.deduplicate_circles."""

    @pytest.mark.parametrize("pass_batch_lengths", [True, False])
    def test_single_batch_item(self, pass_batch_lengths: bool):
        circles = np.array([[0, 0, 1], [0, 0, 0.99]], dtype=np.float64)
        batch_lengths = None
        if pass_batch_lengths:
            batch_lengths = np.array([2], dtype=np.int64)
        deduplication_precision = 1

        deduplicated_circles, deduplicated_batch_lengths, selected_indices = deduplicate_circles(
            circles, deduplication_precision=deduplication_precision, batch_lengths=batch_lengths
        )

        np.testing.assert_array_equal(deduplicated_circles, circles[selected_indices])
        np.testing.assert_array_equal(np.array([1], dtype=np.int64), deduplicated_batch_lengths)

    def test_batch_processing(self):
        circles = np.array(
            [[0, 0, 1], [1, 0, 1], [0, 0, 1.01], [0, 0, 1], [2, 2, 2], [1, 1, 1], [1, 1, 1]], dtype=np.float64
        )
        batch_lengths = np.array([4, 3], dtype=np.int64)

        expected_deduplicated_circles = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        expected_deduplicated_batch_lengths = np.array([2, 2], dtype=np.int64)

        deduplication_precision = 1

        deduplicated_circles, deduplicated_batch_lengths, selected_indices = deduplicate_circles(
            circles, deduplication_precision=deduplication_precision, batch_lengths=batch_lengths
        )

        batch_item_start = 0
        for batch_item_size in expected_deduplicated_batch_lengths:
            current_deduplicated_circles = np.round(
                deduplicated_circles[batch_item_start:batch_item_size], decimals=deduplication_precision
            )
            sorted_indices = np.lexsort(
                (
                    current_deduplicated_circles[:, 2],
                    current_deduplicated_circles[:, 1],
                    current_deduplicated_circles[:, 0],
                )
            )
            np.testing.assert_array_equal(
                expected_deduplicated_circles[batch_item_start:batch_item_size],
                current_deduplicated_circles[sorted_indices],
            )
            batch_item_start += batch_item_size

        np.testing.assert_array_equal(deduplicated_circles, circles[selected_indices])
        np.testing.assert_array_equal(expected_deduplicated_batch_lengths, deduplicated_batch_lengths)

    @pytest.mark.parametrize("deduplication_precision", [1, 4])
    def test_rounding_precision(self, deduplication_precision: int):
        circles = np.array([[0, 0, 1], [0.0001, 0.0001, 0.9999]], dtype=np.float64)
        batch_lengths = np.array([2], dtype=np.int64)

        deduplicated_circles, _, _ = deduplicate_circles(
            circles, deduplication_precision=deduplication_precision, batch_lengths=batch_lengths
        )

        if deduplication_precision < 4:
            assert len(deduplicated_circles) == 1
        else:
            assert len(deduplicated_circles) == len(circles)

    def test_invalid_inputs(self):
        circles = np.zeros((2, 3), dtype=np.float64)
        batch_lengths = np.array([3], dtype=np.int64)

        with pytest.raises(ValueError):
            deduplicate_circles(circles, deduplication_precision=1, batch_lengths=batch_lengths)

    def test_invalid_precision(self):
        circles = np.zeros((2, 3), dtype=np.float64)
        batch_lengths = np.array([2], dtype=np.int64)

        with pytest.raises(ValueError):
            deduplicate_circles(circles, deduplication_precision=-1, batch_lengths=batch_lengths)
