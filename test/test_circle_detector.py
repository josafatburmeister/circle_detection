""" Tests for :code:`circle_detection.CircleDetector`. """

from typing import Any, Dict

import numpy as np
import pytest

from circle_detection import Ransac

from .utils import generate_circle_points


class TestCircleDetector:
    """Tests for :code:`circle_detection.CircleDetector`. Since this is an abstract class, it is tested using the
    derived class :code:`circle_detection.Ransac`"""

    @pytest.mark.parametrize("pass_max_dist", [True, False])
    def test_filtering_circumferential_completeness_index(self, pass_max_dist: bool):
        original_circles = np.array([[0, 0, 0.5]])
        xy = generate_circle_points(original_circles, min_points=100, max_points=100, variance=0)
        bandwidth = 0.01

        max_dist = None
        if pass_max_dist:
            max_dist = bandwidth

        circle_detector = Ransac(bandwidth=bandwidth)
        circle_detector.detect(
            xy,
            num_workers=1,
        )
        circle_detector.filter(
            max_circles=1,
            min_circumferential_completeness_idx=0.9,
            circumferential_completeness_idx_max_dist=max_dist,
            circumferential_completeness_idx_num_regions=int(365 / 5),
            num_workers=1,
        )

        assert len(circle_detector.circles) == 1
        assert len(circle_detector.fitting_scores) == 1

        np.testing.assert_almost_equal(original_circles[0], circle_detector.circles[0], decimal=10)

        circle_detector.detect(
            xy[:50],
            num_workers=1,
        )
        circle_detector.filter(
            max_circles=1,
            min_circumferential_completeness_idx=0.9,
            circumferential_completeness_idx_max_dist=max_dist,
            circumferential_completeness_idx_num_regions=int(365 / 5),
            num_workers=1,
        )

        assert len(circle_detector.circles) == 0
        assert len(circle_detector.fitting_scores) == 0

    def test_filtering_non_maximum_suppression(self):
        original_circles = np.array([[0, 0, 0.5], [0.1, 0.1, 1], [2, 2, 0.5]])
        xy = []
        for circle_idx in range(len(original_circles)):
            xy.append(
                generate_circle_points(
                    original_circles[circle_idx : circle_idx + 1],
                    min_points=100 * circle_idx,
                    max_points=100 * circle_idx,
                    variance=0,
                )
            )
        bandwidth = 0.01

        circle_detector = Ransac(bandwidth=bandwidth)
        circle_detector.detect(
            np.concatenate(xy),
            num_workers=1,
        )
        circle_detector.filter(
            max_circles=2,
            non_maximum_suppression=True,
            num_workers=1,
        )

        expected_circles = np.array([[2, 2, 0.5], [0.1, 0.1, 1]], dtype=np.float64)

        assert len(circle_detector.circles) == 2
        assert len(circle_detector.fitting_scores) == 2
        np.testing.assert_almost_equal(expected_circles, circle_detector.circles)

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

        circle_detector = Ransac(bandwidth=0.01)
        circle_detector.detect(xy)

        with pytest.raises(ValueError):
            circle_detector.filter(**args)

    def test_filtering_without_detection(self):
        circle_detector = Ransac(bandwidth=0.01)
        with pytest.raises(ValueError):
            circle_detector.filter()
