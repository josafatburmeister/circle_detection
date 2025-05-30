"""Abstract base class for implementing circle detectors."""

__all__ = ["CircleDetector"]

import abc
from typing import Any, Optional, cast

import numpy as np

from circle_detection.operations import (
    deduplicate_circles,
    filter_circumferential_completeness_index,
    non_maximum_suppression as non_maximum_suppression_op,
    select_top_k_circles,
)
from circle_detection.type_aliases import FloatArray, LongArray


class CircleDetector(abc.ABC):
    r"""
    Abstract base class for implementing circle detectors.

    Attributes:
        circles: After the :code:`self.detect()` method has been called, this attribute contains the parameters of the
            detected circles (in the following order: x-coordinate of the center, y-coordinate of the center, radius).
            If the :code:`self.detect()` method has not yet been called, this attribute is an empty array.
        fitting_scores: After the :code:`self.detect()` method has been called, this attribute contains the fitting
            scores of the detected circles (higher means better). If the :code:`self.detect()`
            method has not yet been called, this attribute is an empty array.
        batch_lengths_circles: After the :code:`self.detect()` method has been called, this attribute contains the
            number of circles detected for each batch item (circles belonging to the same batch item are stored
            consecutively in :code:`self.circles`). If the :code:`self.detect()` method has not yet been
            called, this attribute is :code:`[0]`.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`fitting_scores`: :math:`(C)`
        - :code:`batch_lengths_circles`: :math:`(B)`

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of detected circles}`
    """

    def __init__(self) -> None:
        self._has_detected_circles = False
        self.circles: FloatArray = np.empty((0, 3), dtype=np.float64)
        self.fitting_scores: FloatArray = np.empty(0, dtype=np.float64)
        self.batch_lengths_circles: LongArray = np.array([0], dtype=np.int64)

        self._xy: FloatArray = np.empty((0, 2), dtype=np.float64)
        self._batch_lengths_xy: LongArray = np.array([0], dtype=np.int64)

    @abc.abstractmethod
    def detect(self, xy: FloatArray, *, batch_lengths: Optional[LongArray] = None, **kwargs: Any):
        """
        This method must be overwritten in subclasses to implement the circle detection approach.

        After successful circle detection, this method should set :code:`self._has_detected_circles` to :code:`True` and
        store the results in :code:`self.circles`, :code:`self.fitting_scores`, and :code:`self.batch_lengths_circles`.
        The corresponding input data must be stored in :code:`self._xy` and :code:`self._batch_lengths_xy`.
        """

    def filter(
        self,
        deduplication_precision: Optional[int] = 4,
        non_maximum_suppression: bool = True,
        max_circles: Optional[int] = None,
        min_circumferential_completeness_idx: Optional[float] = None,
        circumferential_completeness_idx_max_dist: Optional[float] = None,
        circumferential_completeness_idx_num_regions: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        r"""
        Filters the circles whose data are in stored in :code:`self.circles`, :code:`self.fitting_scores`, and
        :code:`self.batch_lengths`, and updates these attributes with the results of the filtering operation. The method
        assumes that :code:`self.detect()` has been called before.

        Args:
            max_circles: Maximum number of circles to return. If more circles are detected, the circles with the lowest
                fitting scores are returned. If set to :code:`None`, all detected circles are returned.
            deduplication_precision: Precision parameter for the deduplication of the circles. If the parameters of two
                detected circles are equal when rounding with the specified numnber of decimals, only one of them is
                kept.
            min_circumferential_completeness_idx: Minimum
                `circumferential completeness index <https://doi.org/10.3390/rs12101652>`__ that a circle must have in
                order to not be discarded. If :code:`min_circumferential_completeness_idx` is set,
                :code:`circumferential_completeness_idx_num_regions` must also be set. If
                :code:`min_circumferential_completeness_idx` is :code:`None`, no filtering based on the circumferential
                completeness index is done.
            circumferential_completeness_idx_max_dist: Maximum distance a point can have to the circle outline to be
                counted as part of the circle when computing the circumferential completeness index. If set to
                :code:`None` and the algorithm has a bandwidth parameter, the bandwidth of the circle detection
                algorithm is used as the default. If the circle detection algorithm has no bandwidth parameter, points
                are counted as part of the circle if their distance to the circle is center is in the interval
                :math:`[0.7 \cdot r, 1.3 \cdot r]` where :math:`r` is the circle radius.
            circumferential_completeness_idx_num_regions: Number of angular regions for computing the circumferential
                completeness index. Must not be :code:`None`, if :code:`min_circumferential_completeness_idx` is not
                :code:`None`.
            non_maximum_suppression: Whether non-maximum suppression should be applied to the detected circles. If this
                option is enabled, circles that overlap with other circles, are only kept if they have the lowest
                fitting loss among the circles with which they overlap.
            num_workers: Number of workers threads to use for parallel processing. If set to -1, all CPU threads are
                used.

        Raises:
            ValueError: if this method is called before calling :code:`self.detect()` or if
                :code:`min_circumferential_completeness_idx` is not :code:`None` and
                :code:`circumferential_completeness_idx_num_regions` is :code:`None`.
        """

        if min_circumferential_completeness_idx is not None and circumferential_completeness_idx_num_regions is None:
            raise ValueError(
                "circumferential_completeness_idx_num_regions must be set if min_circumferential_completeness_idx is "
                + "set."
            )

        if not self._has_detected_circles:
            raise ValueError("The detect() method has to be called before calling the filter() method.")

        if deduplication_precision is not None:
            self.circles, self.batch_lengths_circles, selected_indices = deduplicate_circles(
                self.circles, deduplication_precision, batch_lengths=self.batch_lengths_circles
            )
            self.fitting_scores = self.fitting_scores[selected_indices]

        if non_maximum_suppression and (max_circles is None or max_circles > 1):
            self.circles, self.fitting_scores, self.batch_lengths_circles, _ = non_maximum_suppression_op(
                self.circles, self.fitting_scores, self.batch_lengths_circles, num_workers=num_workers
            )

        if min_circumferential_completeness_idx is not None:
            if circumferential_completeness_idx_max_dist is None:
                if hasattr(self, "_bandwidth"):
                    circumferential_completeness_idx_max_dist = getattr(self, "_bandwidth")
                else:
                    circumferential_completeness_idx_max_dist = -1
            self.circles, self.batch_lengths_circles, selected_indices = filter_circumferential_completeness_index(
                self.circles,
                self._xy,
                num_regions=cast(int, circumferential_completeness_idx_num_regions),
                min_circumferential_completeness_index=min_circumferential_completeness_idx,
                max_dist=circumferential_completeness_idx_max_dist,
                batch_lengths_circles=self.batch_lengths_circles,
                batch_lengths_xy=self._batch_lengths_xy,
                num_workers=num_workers,
            )
            self.fitting_scores = self.fitting_scores[selected_indices]

        if max_circles is not None:
            self.circles, self.fitting_scores, self.batch_lengths_circles, _ = select_top_k_circles(
                self.circles, self.fitting_scores, k=max_circles, batch_lengths=self.batch_lengths_circles
            )
