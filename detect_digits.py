import os

import cv2 as cv
import torch

from models.bbmodel import BBModel
from utils import extract_digits

if __name__ == '__main__':
    weights_path = os.path.join(
        'runs',
        'train',
        'exp',
        'weights',
        'best.pt',
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BBModel(weights_path, device)
    img_path = r'...'
    image = cv.imread(img_path, cv.IMREAD_COLOR)
    bbs = model.detect(image)
    digits = extract_digits(image, bbs)
    for i, digit in enumerate(digits):
        cv.imshow(str(i), digit)
    cv.waitKey()
