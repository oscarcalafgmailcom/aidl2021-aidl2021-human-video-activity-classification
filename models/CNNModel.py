import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNModel(nn.Module):
    def __init__(self, config):

        super().__init__()

        self._h1 = config["h1"]
        self._h2 = config["h2"]
        self._h3 = config["h3"]
        self._h4 = config["h4"]
        self._h5 = config["h5"]
        self._features = config["features"]

        self._imageInputWidth = config["imageInputWidth"]
        self._imageInputHeight = config["imageInputHeight"]
        self._imageResize = config["imageResize"]
        self._imageResizeWidth = config["imageResizeWidth"]
        self._imageResizeHeight = config["imageResizeHeight"]

        self._flatternDimensions = int(
            self._h3 * (self._imageInputWidth * self._imageInputHeight) / (8 * 8)
        )
        if self._imageResize:
            self._flatternDimensions = int(
                self._h3 * (self._imageResizeWidth * self._imageResizeHeight) / (8 * 8)
            )

        self.conv1 = nn.Conv2d(3, self._h1, 3, padding=1)
        self.conv2 = nn.Conv2d(self._h1, self._h2, 3, padding=1)
        self.conv3 = nn.Conv2d(self._h2, self._h3, 3, padding=1)

        # Tinc que trobar una manera automàtica de fer això. 2^pools ^2 dimensions
        self.fc1 = nn.Linear(self._flatternDimensions, self._h4)
        self.fc2 = nn.Linear(self._h4, self._features)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        # x = self.pool(self.vgg16(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
