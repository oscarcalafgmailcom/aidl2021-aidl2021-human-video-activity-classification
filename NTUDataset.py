import pandas as pd
from torch.utils.data import Dataset
import glob
from torchvision.transforms import functional
import cv2
from utils import resize, addMomentToTensor


class NTUDataset(Dataset):
    def __init__(self, criteria, config):
        self.criteria = criteria
        self.files = []
        for crite in criteria:
            self.files.extend(glob.glob(crite))
        self.config = config
        self.imageResize = config["imageResize"]
        self.imageResizeWidth = config["imageResizeWidth"]
        self.imageResizeHeight = config["imageResizeHeight"]
        self.frameRate = config["frameRate"]
        self.labels = []
        for f in self.files:
            self.labels.append(f[len(f) - 12 : -8])

        self.labels.sort()
        self.labels = list(set(self.labels))
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = path[len(path) - 12 : -8]

        classification = self.labels.index(label)
        return path, classification
