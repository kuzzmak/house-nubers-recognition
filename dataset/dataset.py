from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset as Dat
from torchvision import transforms

from bounding_box import BoundingBox
from utils import get_file_paths_from_dir

# if no transformation are passed on class initialization, this default
# transformation is used
default_transforms = transforms.Compose([transforms.ToTensor()])


@dataclass
class _Label:
    """Helper class containing bounding boxes of digits on the image and
    corresponding numbers.
    """
    numbers: List[int]
    bounding_boxes: List[BoundingBox]


class Dataset(Dat):

    def __init__(
        self,
        data_path: str,
        target_size: int = 128,
        device: str = 'cpu',
        transforms: Optional[transforms.Compose] = None,
    ):
        """Simple dataset for the recognition of the house numbers.

        Parameters
        ----------
        data_path : str
            path to the folder with images
        target_size : int
            size of the square of the image once resized
        device : str, optional
            device on which data will be loaded, by default 'cpu'
        transforms : Optional[transforms.Compose], optional
            transformations for the data, by default None
        """
        self.data_path = data_path
        self.target_size = target_size
        self.device = device
        self.transforms = transforms if transforms is not None \
            else default_transforms
        print('Loading data...')
        self._data_paths = get_file_paths_from_dir(data_path, ['png'])
        print('Data loaded.')
        print('Loading labels...')
        self._labels = self._load_labels()
        print('Labels loaded.')

    def _load_labels(self) -> Dict[str, _Label]:
        """Loads labels of the `train` or `test` datasets into a dictionary
        each key is the image name like `10.png` and each value is an object
        of type `Label` which contains bounding boxes around each digit on
        the image and a corresponding number for each digit in range 0..10.

        Returns
        -------
        Dict[str, Label]
            dictionary of labels
        """
        labels_dict = dict()
        labels_path = os.path.join(self.data_path, 'labels.txt')
        with open(labels_path, 'r') as f:
            while True:
                # remove newlines
                row = f.readline().strip()
                if not row:
                    break
                parts = row.split(';')
                # remove empty strings
                parts = [p for p in parts if len(p) > 0]
                img_name = parts.pop(0)
                bbs = []
                numbers = []
                for part in parts:
                    split = part.split(',')
                    split = list(map(int, split))
                    bb = BoundingBox(*split[:4])
                    numbers.append(split[-1])
                    bbs.append(bb)
                labels_dict[img_name] = _Label(numbers, bbs)
        return labels_dict

    def _load_from_path(self, path: str, flags=cv.IMREAD_COLOR) -> np.ndarray:
        """Loads image from the path as a `np.ndarray`. 

        NOTE: OpenCV loads images in BGR format, and not in RGB format.

        Parameters
        ----------
        path : str
            path to the image
        flags : [int], optional
            flags for the imread function, by default cv.IMREAD_COLOR

        Returns
        -------
        np.ndarray
            image as an `np.ndarray`
        """
        image = cv.imread(path, flags)
        return image

    def _pad(
        self,
        image: np.ndarray,
        padding_color=[0, 0, 0],
    ) -> Tuple[np.ndarray, float, int, int, int, int]:
        """Pads image with desired color so the resulting image is of the
        square shape.

        Parameters
        ----------
        image : np.ndarray
            image to pad
        target_size : int, optional
            size of the square image, by default 128
        padding_color : list, optional
            padding color, by default [0, 0, 0] (black)

        Returns
        -------
        Tuple[np.ndarray, float, int, int, int, int]
            padded image, ratio used for scaling of the bonding box, paddings
            for the every side of the image
        """
        old_size = image.shape[:2]
        ratio = float(self.target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        image = cv.resize(image, (new_size[1], new_size[0]))
        delta_w = self.target_size - new_size[1]
        delta_h = self.target_size - new_size[0]
        top, bottom = delta_h//2, delta_h - delta_h//2
        left, right = delta_w//2, delta_w - delta_w//2
        image = cv.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv.BORDER_CONSTANT,
            value=padding_color,
        )
        return image, ratio, top, bottom, left, right

    def _bounding_boxes_to_tensor(
        self, 
        bounding_boxes: List[BoundingBox],
    ) -> torch.Tensor:
        """Converts list of bounding boxes into tensor.

        Parameters
        ----------
        bounding_boxes : List[BoundingBox]
            bounding boxes to convert

        Returns
        -------
        torch.Tensor
            tensor of bounding boxes
        """
        boxes = [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bounding_boxes]
        return torch.Tensor(boxes)

    def _numbers_to_tensor(self, numbers: List[int]) -> torch.Tensor:
        """Converts list of numbers into tensor.

        Parameters
        ----------
        numbers : List[int]
            list of numbers to convert

        Returns
        -------
        torch.Tensor
            tensor of numbers
        """
        numbers = [[num] for num in numbers]
        return torch.Tensor(numbers)

    def __len__(self):
        return len(self._data_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, _Label]:
        # returning a tensor which is a image, tensor of bounding boxes around
        # the digits on the image and tensor of digits on the image (may change
        # in future to single digit)
        path = self._data_paths[index]
        image = self._load_from_path(path)
        image, ratio, top, bottom, left, right = self._pad(image)

        img_name = os.path.basename(path)
        label: _Label = self._labels[img_name]

        [bb.transform(ratio, top, bottom, left, right) for bb in label.bounding_boxes]

        bounding_boxes = self._bounding_boxes_to_tensor(label.bounding_boxes)
        numbers = self._numbers_to_tensor(label.numbers)

        image = self.transforms(image)
        return image, bounding_boxes, numbers
