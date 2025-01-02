""" Non-maximum suppression operation to remove overlapping circles. """

__all__ = ["non_maximum_suppression"]

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from circle_detection.operations._operations_cpp import (  # type: ignore[import-not-found] # pylint: disable=import-error, no-name-in-module
    non_maximum_suppression as non_maximum_suppression_cpp,
)


def non_maximum_suppression(
    circles: npt.NDArray[np.float64],
    fitting_losses: npt.NDArray[np.float64],
    batch_lengths: Optional[npt.NDArray[np.int64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    r"""
    Non-maximum suppression operation to remove overlapping circles. If a circle overlaps with other circles, it is
    only kept if it has the lowest fitting loss among the circles with which it overlaps. This method supports batch
    processing, i.e. separate sets of circles (i.e., different batch items) can be processed in parallel. For this
    purpose, :code:`batch_lengths` must be set to specify which circle belongs to which set.

    Args:
        circles: Parameters of the circles to which apply non-maximum suppression (in the following order:
            x-coordinate of the center, y-coordinate of the center, radius).
        fitting_losses: Fitting losses of the circles to which apply non-maximum suppression (lower means better).
        batch_lengths: Number of circles in each item of the input batch. For batch processing, it is expected that
            all circles and fitting losses belonging to the same batch item are stored consecutively in the respective
            input array. For example, if a batch comprises two batch items with :math:`N_1` circles and :math:`N_2`
            circles, then :code:`batch_lengths` should be set to :code:`[N_1, N_2]` and :code:`circles[:N_1]` should
            contain the circles of the first batch item and :code:`circles[N_1:]` the circles of the second batch item.
            If :code:`batch_lengths` is set to :code:`None`, it is assumed that the input circles belong to a single
            batch item and batch processing is disabled. Defaults to :code:`None`.

    Returns:
        : Tuple of four arrays. The first contains the parameters of the circles remaining after non-maximum
        suppression and the second the corresponding fitting losses. The third contains the number of circles in each
        item of the output batch. The fourth contains the indices of the selected circles in the input array.

    Raises:
        ValueError: If :code:`circles` and :code:`fitting_losses` have different lengths or if :code:`batch_lengths` is
            not :code:`None` and the length of :code:`circles` is not equal to the sum of :code:`batch_lengths`.

    Shape:
        - :code:`circles`: :math:`(C, 3)`
        - :code:`fitting_losses`: :math:`(C)`
        - :code:`batch_lengths`: :math:`(B)`
        - Output: The first array in the output tuple has shape :math:`(C', 3)`, the second shape :math:`(C')`, the
          third shape :math:`(B)`, and the fourth has shape :math:`(C')`.

        | where
        |
        | :math:`B = \text{ batch size}`
        | :math:`C = \text{ number of circles before applying non-maximum suppression}`
        | :math:`C' = \text{ number of circles after applying non-maximum suppression}`
    """
    if batch_lengths is None:
        batch_lengths = np.array([len(circles)], dtype=np.int64)

    return non_maximum_suppression_cpp(circles, fitting_losses, batch_lengths)
