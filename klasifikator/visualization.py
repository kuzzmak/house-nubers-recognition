import torch
from klasifikator import model
from models.bbmodel import BBModel
import cv2 as cv
from torchvision import transforms
from utils import extract_digits
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def draw_box(boxes, img, prediction, index):
    fig, ax = plt.subplots()

    ax.imshow(img)
    for box in boxes:
        rect = patches.Rectangle((box.x1, box.y1), (box.x2-box.x1), (box.y2-box.y1), linewidth=1, edgecolor='r', fill=False)
        ax.add_patch(rect)

    if prediction != '':
        plt.title("Prediction: " + str(prediction))
    picture_index = './results/' + str(index) + '.png'
    plt.savefig(picture_index)


bounding_weights_path = r'../runs/train/exp/weights/best.pt'
classif_weights_path = r'./weights/model.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
bounding_model = BBModel(bounding_weights_path, device)

classif_model = model.resnet18()
classif_model.load_state_dict(torch.load(classif_weights_path))
classif_model.eval()

for i in range(0, 50):
    img_path = '../data/test/'+str(i+1)+'.png'
    slika = Image.open(img_path)
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
    draw_box(bbs, slika, string, i+1)