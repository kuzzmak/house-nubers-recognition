import numpy as np

import torch

from bounding_box import BoundingBox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)


class BBModel:

    def __init__(self, weights_path: str, device: str = 'cpu'):
        self._weights_path = weights_path
        self._device = device
        self._model = self._load_model()

    def _load_model(self):
        model = DetectMultiBackend(
            self._weights_path,
            device=self._device,
        )
        return model

    @torch.no_grad()
    def detect(
        self,
        input_image: np.ndarray,
        imgsz=[150, 150],
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
    ):
        stride = 32
        imgsz = check_img_size(imgsz, s=stride)

        padded_image = letterbox(
            input_image,
            imgsz,
            stride=stride,
            auto=False,
        )[0]
        padded_image = padded_image.transpose((2, 0, 1))[::-1]
        padded_image = np.ascontiguousarray(padded_image)
        # self._model.warmup(imgsz=(1, 3, *imgsz), half=False)

        normalized_image = torch.from_numpy(padded_image).to(self._device)
        normalized_image = normalized_image.float()
        normalized_image /= 255
        if len(normalized_image.shape) == 3:
            normalized_image = normalized_image[None]
        pred = self._model(normalized_image, augment=False, visualize=False)

        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            None,
            False,
            max_det=max_det,
        )

        bbs = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(
                    normalized_image.shape[2:],
                    det[:, :4],
                    input_image.shape,
                ).round()
                det = det.cpu().numpy()[:, :4].tolist()
                for d in det:
                    d = list(map(int, d))
                    bbs.append(d)
                bbs = sorted(bbs, key=lambda x: (x[0], x[1]))
                bbs = list(map(lambda x: BoundingBox(*x), bbs))
        return bbs
