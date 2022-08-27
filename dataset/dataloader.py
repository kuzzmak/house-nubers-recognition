from typing import List, Tuple

import torch
from torch.utils.data import DataLoader as DL
from torch.utils.data.dataset import Dataset


def _pad(
    data: torch.Tensor,
    input_size: int,
    max_batch_length: int,
) -> torch.Tensor:
    """Function for padding examples in a single batch. This is needed in
    order for `DataLoader` to work properly. `DataLoader` must have all
    tensors of the same shape in the batch.

    Parameters
    ----------
    data : torch.Tensor
        batch that is being padded
    input_size : int
        "width" of overy example, 4 for bounding box and 1 for number
    max_batch_length : int
        biggest number of rows any example in batch has, used for appending
        dummy values so in the end, every tensor has the same shape

    Returns
    -------
    torch.Tensor
        padded batch
    """
    new_data = []
    # go through every example in batch
    for i, val in enumerate(data):
        # check if it's needed to pad this example (if so, his first
        # dimension is smaller than it's supposed to be)
        tensor_rows = val.size()[0]
        if tensor_rows != max_batch_length:
            old_tensor = val
            # add dummy rows to satisfy shape of the target tensor
            for _ in range(max_batch_length - tensor_rows):
                # -1 is used as a dummy value so if bounding boxes are
                # being padded, row looking like [-1, -1, -1, -1] is being
                # added and if it's number that is being padded, only [-1]
                # is appended to the tensor for how many times it's needed
                old_tensor = torch.cat(
                    (old_tensor, torch.Tensor([[-1] * input_size])),
                    dim=0,
                )
            new_data.append(old_tensor)
        else:
            # no padding
            new_data.append(val)
    # make a tensor from the list
    new_data = torch.stack(new_data)
    return new_data


def unpad(data: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
    """Function to remove padding from the batch so only useful data remains.

    Parameters
    ----------
    data : torch.Tensor
        padded batch
    lengths : torch.Tensor
        tensor containing real data length for every example, using list because
        it's not possible to construct a tensor where some dimensions are of a 
        different size

    Returns
    -------
    List[torch.Tensor]
        list of unpadded tensors
    """
    real_data = [torch.narrow(row, 0, 0, lengths[i])
                 for i, row in enumerate(data)]
    return real_data


def collate_fn(
    batch: List[torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Function used by `DataLoader` to correctly structure data so it can be
    used normally. That includes padding examples in a batch so every batch
    has tensors of the same shape.

    Parameters
    ----------
    batch : List[torch.Tensor]
        batch of data received from `Dataset`

    Returns
    -------
    Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        batch and real data length of every example in this batch
    """
    images = [elem[0] for elem in batch]
    bbs = [elem[1] for elem in batch]
    numbers = [elem[2] for elem in batch]
    lengths = torch.tensor([len(t) for t in bbs])
    max_length = lengths.max()
    padded_bbs = _pad(bbs, 4, max_length)
    padded_numbers = _pad(numbers, 1, max_length)
    return (torch.stack(images), padded_bbs, padded_numbers), lengths


class DataLoader(DL):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        """Class for menaging of the data batches, shuffling of the data and
        iteration through data.

        Parameters
        ----------
        dataset : Dataset
            dataset
        batch_size : int
            how many examples in one batch of data
        shuffle : bool, optional
            should data be shuffled, by default True
        """
        super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn)
