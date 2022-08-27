import os
import torch
from models.bbmodel import BBModel
import cv2 as cv
from utils import extract_digits
from torchvision import transforms
from PIL import Image
import numpy as np
import model

labels_file = open(r'../data/test/labels.txt', 'r')
lines = labels_file.readlines()
labels = []

for line in lines:
    data = line.split(';')
    tmp = ""
    for i in range(1, len(data)-1):
        label = data[i].split(',')
        if label[4] == '10':
            label[4] = '0'
        tmp = tmp + str(label[4])
    labels.append(tmp)


bounding_weights_path = r'../runs/train/exp/weights/best.pt'
classif_weights_path = r'./weights/model.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bounding_model = BBModel(bounding_weights_path, device)

classif_model = model.resnet18()
classif_model.load_state_dict(torch.load(classif_weights_path))
classif_model.eval()

for i in range(0, 1000):
    img_path = '../data/test/'+str(i+1)+'.png'
    image = cv.imread(img_path, cv.IMREAD_COLOR)
    bbs = bounding_model.detect(image)
    digits = extract_digits(image, bbs)
    p = transforms.Compose([transforms.Resize((32, 32))])
    softmax = torch.nn.Softmax(dim=1)
    predictions = []
    for j, digit in enumerate(digits):
        im = Image.fromarray(np.uint8(digit)).convert('RGB')
        im = p(im)
        im = torch.tensor(np.array(im))
        im = im[None,:,:,:]
        im = torch.reshape(im, (1, 3, 32, 32))
        im = im.float()
        output = classif_model(im)
        output = softmax(output)
        prediction = torch.argmax(output)
        predictions.append(prediction)

    string = ""
    for prediction in predictions:
        string += str(prediction.item())

    if string == '':
        print("Prediction for image %d is: " % (i+1))
    else:
        print("Prediction for image %d is: %d" % (i+1, int(string)))
    print("Label for image %d is: %d" % (i+1, int(labels[i])))


