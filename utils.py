from typing import List, Optional
import os

import cv2 as cv
import numpy as np
import torch

from bounding_box import BoundingBox


def get_file_extension(file_path: str) -> str:
    """Gets file extension.

    Parameters
    ----------
    file_path : str
        path of the file

    Returns
    -------
    str
        file extension
    """
    ext = file_path.split('.')[-1]
    return ext


def get_file_paths_from_dir(
    dir: str,
    extensions: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Constructs apsolute file paths of the files in `dir`. If files
    with particular extensions are allowed, then `extensions` argument
    should be also passed to function.

    Parameters
    ----------
    dir : str
        directory with files
    extensions : Optional[List[str]], optional
        files that end with these extension will be included, by default None

    Returns
    -------
    Optional[List[str]]
        list of file paths is they satisfy `extensions` argument
    """
    if not os.path.exists(dir):
        return None

    files = [f for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    curr_dir = os.path.abspath(dir)
    file_paths = [os.path.join(curr_dir, x) for x in files]

    if extensions is None:
        return file_paths

    exts = set(extensions)
    file_paths = [f for f in file_paths if get_file_extension(f) in exts]

    return file_paths


def display_bounding_boxes(image: np.ndarray, bbs: List[BoundingBox]) -> None:
    """Displays bounding boxes on the image.

    Parameters
    ----------
    image : np.ndarray
        iamge where bounding boxes are drawn
    bbs : List[BoundingBox]
        bounding boxes to draw
    """
    for bb in bbs:
        image = cv.rectangle(
            image,
            (bb.x1, bb.y1),
            (bb.x2, bb.y2),
            (255, 0, 0),
            2,
        )
    cv.imshow('im', image)
    cv.waitKey(0)


def tensor_to_np_image(image: torch.Tensor) -> np.ndarray:
    """Converts image in a form of a `torch.Tensor` into image in `np.ndarray`
    format. In tensor form, image is in 0..1 range so it has to be multiplied
    by 255 in order to be displayed correctly in `Figure`.

    Args:
        image (torch.Tensor): image in tensor form

    Returns:
        np.ndarray: image in numpy form
    """
    img = image.cpu().detach().numpy()
    img = img * 255
    # move number of channels to the last dimension
    img = img.transpose(1, 2, 0)
    img = np.float32(img)
    # image was initially in BGR format, convert to RGB to properly show in
    # maptlotlib canvas
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def tensor_to_bounding_box(bb: torch.Tensor) -> BoundingBox:
    bb = bb.cpu().detach().numpy().astype(np.uint8)
    bb = bb.tolist()
    return BoundingBox(*bb)


def extract_digits(
    image: np.ndarray,
    bbs: List[BoundingBox],
) -> List[np.ndarray]:
    """Extracts parts of the image defined by bounding boxes.

    Args:
        image (np.ndarray): image with digits
        bbs (List[BoundingBox]): list of bounding boxes defining every digit
            on image

    Returns:
        List[np.ndarray]: parts of image defined by bounding boxes
    """
    digits = [image[bb.y1:bb.y2, bb.x1:bb.x2] for bb in bbs]
    return digits
