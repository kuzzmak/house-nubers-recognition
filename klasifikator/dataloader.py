import torch
import scipy.io
from torch.utils.data import Dataset
from torchvision import datasets


class CustomDataset(Dataset):

    def __init__(self, img_dir):
        self.mat = scipy.io.loadmat(img_dir)
        self.X = self.mat['X']
        self.y = self.mat['y']
        self.y[self.y==10] = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[:,:,:,idx], self.y[idx]