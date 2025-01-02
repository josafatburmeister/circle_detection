""" Deduplication of circles. """

__all__ = ["deduplicate_circles"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def deduplicate_circles(
    circles: npt.NDArray[np.float64],
    deduplication_precision: int,
    batch_lengths: Optional[npt.NDArray[np.int64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Deduplicates circles whose parameters do not differ up to the decimal place specified by
    :code:`deduplication_precision`. This method supports batch processing, i.e. separate sets of
    circles (i.e., different batch items) can be processed in parallel. For this purpose, :code:`batch_lengths` must be
    set to specify which circle belongs to which set.

    Args:
        circles: Parameters of the circles to deduplicate (in the following order: x-coordinate of the center,
            y-coordinate of the center, radius).
        deduplication_precision: Number of decimal places taken into account for deduplication.
        batch_lengths: Number of circles in each item of the input batch. For batch processing, it is expected that
            all circles and fitting losses belonging to the same batch item are stored consecutively in the respective
            input array. For example, if a batch comprises two batch items with :math:`N_1` circles and :math:`N_2`
            circles, then :code:`batch_lengths` should be set to :code:`[N_1, N_2]` and :code:`circles[:N_1]` should
            contain the circles of the first batch item and :code:`circles[N_1:]` the circles of the second batch item.
            If :code:`batch_lengths` is set to :code:`None`, it is assumed that the input circles belong to a single
            batch item and batch processing is disabled. Defaults to :code:`None`.

    Returns:
        : Tuple of three arrays: The first contains the parameters of the circles remaining after deduplication. The
        second contains the number of circles in each item of the output batch. The third contains the indices of the
        selected circles in the input array.

    Raises:
        ValueError: If :code:`deduplication_precision` is not a positive number or :code:`batch_lengths` is not
            :code:`None` and the length of :code:`circles` is not equal to the sum of :code:`batch_lengths`.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`batch_lengths`: :math:`(B)`
        - Output: The first array in the output tuple has shape :math:`(C', 3)`, the second shape :math:`(B)`, and the
          third shape :math:`(C')`.

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of circles before deduplication}`
        | :math:`C' = \text{ number of circles after deduplication}`
    """

    if deduplication_precision < 0:
        raise ValueError("deduplication_precision must be a positive number.")

    if batch_lengths is not None and len(circles) != batch_lengths.sum():
        raise ValueError("The number of circles must be equal to the sum of batch_lengths.")

    if batch_lengths is None or len(batch_lengths) == 1:
        rounded_circles = np.round(circles, decimals=deduplication_precision)

        unique_rounded_circles, selected_indices = np.unique(rounded_circles, return_index=True, axis=0)

        return circles[selected_indices], np.array([len(unique_rounded_circles)], dtype=np.int64), selected_indices

    # add batch indices as first dimension to separate circles from different batch items
    rounded_circles = np.empty((len(circles), 4), dtype=np.int64)
    rounded_circles[:, 0] = np.repeat(np.arange(len(batch_lengths), dtype=np.float64), batch_lengths)
    rounded_circles[:, 1:] = np.round(circles, decimals=deduplication_precision)

    unique_rounded_circles, selected_indices = np.unique(rounded_circles, return_index=True, axis=0)

    batch_item_borders_mask = np.empty(len(unique_rounded_circles), dtype=bool)
    batch_item_borders_mask[:1] = True
    batch_item_borders_mask[1:] = unique_rounded_circles[1:, 0] != unique_rounded_circles[:-1, 0]
    deduplicated_batch_lengths = np.zeros(len(batch_lengths), dtype=np.int64)
    deduplicated_batch_lengths[batch_lengths > 0] = np.diff(
        np.concatenate(np.nonzero(batch_item_borders_mask) + ([len(batch_item_borders_mask)],))
    )

    return circles[selected_indices], deduplicated_batch_lengths, selected_indices
