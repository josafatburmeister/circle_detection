""" Selection of circles with the lowest fitting losses. """

__all__ = ["select_top_k_circles"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def select_top_k_circles(
    circles: npt.NDArray[np.float64],
    fitting_losses: npt.NDArray[np.float64],
    k: int,
    batch_lengths: Optional[npt.NDArray[np.int64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Selects the :code:`k` circles with the lowest fitting losses from a set of circles. If the set of circles contains
    less than :code:`k` circles, all circles are kept. This method supports batch processing, i.e. separate sets of
    circles (i.e., different batch items) can be filtered in parallel. For this purpose, :code:`batch_lengths` must be
    set to specify which circle belongs to which set.

    Args:
        circles: Parameters of the circles to filter (in the following order: x-coordinate of the center, y-coordinate
            of the center, radius).
        fitting_losses: Fitting losses of the circles to filter (lower means better).
        k: Number of circles to select from each set of circles.
        batch_lengths: Number of circles in each item of the input batch. For batch processing, it is expected that
            all circles and fitting losses belonging to the same batch item are stored consecutively in the respective
            input array. For example, if a batch comprises two batch items with :math:`N_1` circles and :math:`N_2`
            circles, then :code:`batch_lengths` should be set to :code:`[N_1, N_2]` and :code:`circles[:N_1]` should
            contain the circles of the first batch item and :code:`circles[N_1:]` the circles of the second batch item.
            If :code:`batch_lengths` is set to :code:`None`, it is assumed that the input circles belong to a single
            batch item and batch processing is disabled. Defaults to :code:`None`.

    Returns:
        : Tuple of four arrays: The first contains the parameters of the selected circles and the second the
        corresponding fitting losses. The third contains the number of circles in each item of the output batch. The
        fourth contains the indices of the selected circles in the input array.

    Raises:
        ValueError: If :code:`circles` and :code:`fitting_losses` have different lengths or if :code:`batch_lengths: is
            not :code:`None` and the length of :code:`circles` is not equal to the sum of :code:`batch_lengths`.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`fitting_losses`: :math:`(C)`
        - :code:`batch_lengths`: :math:`(B)`
        - Output: The first array in the output tuple has shape :math:`(C', 3)`, the second shape :math:`(C')`, the
          third shape :math:`(B)`, and the fourth shape :math:`(C')`.

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of circles before filtering}`
        | :math:`C' = \text{ number of circles after filtering}`
    """
    if len(circles) != len(fitting_losses):
        raise ValueError("circles and fitting_losses must have the same number of entries.")

    if batch_lengths is not None and len(circles) != batch_lengths.sum():
        raise ValueError("The number of circles must be equal to the sum of batch_lengths.")

    if batch_lengths is None or len(batch_lengths) == 1:
        sorting_indices = np.argsort(fitting_losses)

        batch_lengths = np.array([min(k, len(circles))], dtype=np.int64)
        selected_indices = np.arange(len(circles), dtype=np.int64)[sorting_indices][:k]

        return circles[sorting_indices][:k], fitting_losses[sorting_indices][:k], batch_lengths, selected_indices

    batch_indices = np.repeat(np.arange(len(batch_lengths), dtype=np.float64), batch_lengths)
    sorting_indices = np.lexsort((fitting_losses, batch_indices))
    selected_indices = np.cumsum(np.concatenate(([0], batch_lengths)))[:-1]

    offsets = np.arange(k, dtype=np.int64)
    selected_indices = selected_indices[:, None] + offsets
    valid_mask = offsets[None, :] < batch_lengths[:, None]

    selected_indices = selected_indices[valid_mask]
    selected_indices = np.arange(len(circles), dtype=np.int64)[sorting_indices][selected_indices]
    selected_indices = np.sort(selected_indices)

    filtered_batch_lengths = np.full(len(batch_lengths), fill_value=k, dtype=np.int64)
    filtered_batch_lengths = np.minimum(filtered_batch_lengths, batch_lengths)  # type: ignore[assignment]

    return circles[selected_indices], fitting_losses[selected_indices], filtered_batch_lengths, selected_indices
