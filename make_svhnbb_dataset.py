import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm


def get_data_path(data_type: str) -> str:
    """Constructs path to the data based on the type of data `data_type`.

    Parameters
    ----------
    data_type : str
        type of data for which path is being constructed

    Returns
    -------
    str
        path to data
    """
    return os.path.join('data', data_type)


def get_labels_path(data_type: str) -> str:
    """Constructs path to the labels file for specific type of data
    `data_type`.

    Parameters
    ----------
    data_type : str
        type of data fowr which labels path is being constructed

    Returns
    -------
    str
        path to labels file
    """
    return os.path.join(get_data_path(data_type), 'labels.txt')


def get_image_names(data_type: str) -> List[str]:
    """Gets all filenames for a particular data type that have `png` extension.

    Parameters
    ----------
    data_type : str
        type of data for which image names are collected

    Returns
    -------
    List[str]
        list of all image names that end with `png`
    """
    data_path = get_data_path(data_type)
    image_names = os.listdir(data_path)
    image_names = [img for img in image_names if img.endswith('png')]
    return image_names


def load_data(
    data_type: str,
    data_size: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """Loads images for a particular type of data `data_type`.

    Parameters
    ----------
    data_type : str
        type of data for which images will be loaded
    data_size : Optional[int], optional
        how much data should be loaded, if `None`, everything will be loaded,
            by default None
    shuffle : bool, optional
        should images be shuffled, by default True

    Returns
    -------
    Tuple[List[np.ndarray], List[str]]
        list of images, list of image names
    """
    data_path = get_data_path(data_type)
    image_names = get_image_names(data_type)
    if shuffle:
        random.shuffle(image_names)
    if data_size is not None:
        image_names = image_names[:data_size]
    image_paths = [
        os.path.join(data_path, img_name) for img_name in image_names
    ]
    images = np.array(
        [
            cv.imread(path, cv.IMREAD_COLOR) for path in image_paths
        ],
        dtype=object,
    )
    return images, image_names


def load_labels(data_type: str) -> Dict[str, List[str]]:
    """Loads labels for a particular type of data `data_type`.

    Parameters
    ----------
    data_type : str
        type of data for which labels are being loaded

    Returns
    -------
    Dict[str, List[str]]
        dictionary where key is the image name and value is a list of strings
            representing every bounding box
    """
    raw_labels_path = get_labels_path(data_type)
    with open(raw_labels_path, 'r') as f:
        raw_labels = f.readlines()
    raw_labels = [row.split(';') for row in raw_labels]
    raw_labels_keys = list(map(lambda x: x[0], raw_labels))
    raw_labels_values = list(map(lambda x: x[1:-1], raw_labels))
    raw_labels = {
        key: value for (key, value) in zip(raw_labels_keys, raw_labels_values)
    }
    return raw_labels


def make_svhnbb_dataset_directories(data_type: str) -> Tuple[str, str]:
    """Makes folders if they already do not exist for SVHN bounding box
    dataset. After running this function following folder structure is
    made.

    - datasets
        - SVHNBB
            - images
                - train
                - val
            - labels
                - train
                - val

    This folder structure is required by YOLO model.

    Parameters
    ----------
    data_type : str
        type of data for which directories are being made

    Returns
    -------
    Tuple[str, str]
        path to images, path to labels
    """
    svhnbb_dataset_path = os.path.join('datasets', 'SVHNBB')
    svhnbb_images_path = os.path.join(
        svhnbb_dataset_path,
        'images',
        data_type,
    )
    if not os.path.exists(svhnbb_images_path):
        os.makedirs(svhnbb_images_path)
    svhnbb_labels_path = os.path.join(
        svhnbb_dataset_path,
        'labels',
        data_type,
    )
    if not os.path.exists(svhnbb_labels_path):
        os.makedirs(svhnbb_labels_path)
    return svhnbb_images_path, svhnbb_labels_path


def make_dataset(
    images: List[np.ndarray],
    data_type: str,
    image_names: List[str],
    svhnbb_labels_path: str,
    svhnbb_images_path: str,
    raw_labels: Dict[str, List[str]],
    new_shape: Optional[Tuple[int, int]] = None,
) -> None:
    """Utility function for resizing existing pictures and copying them to
    correct folder along with labels that represent bounding boxes on every
    image.

    Parameters
    ----------
    images : List[np.ndarray]
        list of images to copy and resize
    data_type : str
        type of data that is being processed
    image_names : List[str]
        list of image names that correspond to `images` argument
    svhnbb_labels_path : str
        directory for saving labels
    svhnbb_images_path : str
        directory for saving resized images
    raw_labels : Dict[str, List[str]]
        directory of labels which are being saved
    new_shape : Optional[Tuple[int, int]], optional
        shape of new resized images, by default None
    """
    pbar = tqdm(images, f'{data_type} images done: ')
    for i, img in enumerate(pbar):
        label_filename = image_names[i].split('.')[0] + '.txt'
        with open(os.path.join(svhnbb_labels_path, label_filename), 'w') as f:
            if new_shape is not None:
                resized = cv.resize(img, new_shape, cv.INTER_CUBIC)
            else:
                resized = img
            cv.imwrite(
                os.path.join(
                    svhnbb_images_path,
                    image_names[i],
                ),
                resized,
            )
            img_name = image_names[i]
            bbs = raw_labels[img_name]
            h, w, *_ = img.shape
            new_img_shape = tuple(resized.shape[:2])
            for bb in bbs:
                split = bb.split(',')[:4]
                x1, y1, x2, y2 = list(map(lambda x: int(x), split))
                mask = np.zeros((h, w))
                mask[y1:y2, x1:x2] = 1
                if new_shape is not None:
                    mask = cv.resize(mask, new_shape, cv.INTER_CUBIC)
                cols, rows = np.nonzero(mask)
                if len(cols) == 0 or len(rows) == 0:
                    continue
                x1 = np.min(rows)
                y1 = np.min(cols)
                x2 = np.max(rows)
                y2 = np.max(cols)
                c_x = ((x2 + x1) / 2) / new_img_shape[1]
                c_y = ((y2 + y1) / 2) / new_img_shape[0]
                w_scaled = (x2 - x1) / new_img_shape[1]
                h_scaled = (y2 - y1) / new_img_shape[0]
                save_string = f'0 {str(c_x)} {str(c_y)} {str(w_scaled)} ' + \
                    f'{str(h_scaled)}\n'
                f.write(save_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make SVHN bounding boxes dataset.'
    )
    parser.add_argument(
        '-image_shape',
        nargs='+',
        type=int,
        help='shape of each image',
    )
    parser.add_argument(
        '-data_size',
        type=int,
        help='size of the dataset',
    )
    args = parser.parse_args()

    image_shape = args.image_shape
    if image_shape is not None:
        image_shape = tuple(image_shape)

    data_type = 'train'
    data_size = args.data_size
    if data_size is None:
        print('Using all of the available data.')
    print('Loading data, please wait...')
    images, image_names = load_data(data_type, data_size)
    print(f'Loaded {len(images)} images.')
    svhnbb_images_path, svhnbb_labels_path = make_svhnbb_dataset_directories(
        data_type
    )
    raw_labels = load_labels(data_type)

    data_size = len(images)
    train_data_size = int(0.9 * data_size)
    print(f'Images for training: {train_data_size}.')
    train_images = images[:train_data_size]
    train_image_names = image_names[:train_data_size]
    make_dataset(
        train_images,
        data_type,
        train_image_names,
        svhnbb_labels_path,
        svhnbb_images_path,
        raw_labels,
        image_shape,
    )

    data_type = 'val'
    svhnbb_images_path, svhnbb_labels_path = make_svhnbb_dataset_directories(
        data_type
    )
    val_images = images[train_data_size:]
    print(f'Images for validation: {len(val_images)}.')
    val_image_names = image_names[train_data_size:]
    make_dataset(
        val_images,
        data_type,
        val_image_names,
        svhnbb_labels_path,
        svhnbb_images_path,
        raw_labels,
        image_shape,
    )
