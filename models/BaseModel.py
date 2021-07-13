import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, config, device):

        super().__init__()

        self._config = config
        self._h1 = config["h1"]
        self._h2 = config["h2"]
        self._h3 = config["h3"]
        self._h4 = config["h4"]
        self._h5 = config["h5"]
        self._batchSize = config["batch_size"]
        self._features = config["features"]
        self._minFrames = config["minFrames"]
        self._dropOut = config["dropOut"]

        self._imageChannels = config["imageChannels"]
        self._imageInputWidth = config["imageInputWidth"]
        self._imageInputHeight = config["imageInputHeight"]
        self._imageResize = config["imageResize"]
        self._imageResizeWidth = config["imageResizeWidth"]
        self._imageResizeHeight = config["imageResizeHeight"]

        self._imageWidth = self._imageInputWidth
        self._imageHeight = self._imageInputHeight
        if self._imageResize:
            self._imageWidth = self._imageResizeWidth
            self._imageHeight = self._imageResizeHeight

        self._flatternDimensions = int(
            self._h3 * (self._imageWidth * self._imageHeight) / (8 * 8)
        )

        self._hidden = None
        self._device = device


    def dropout(self, x):
        return F.dropout(x, p=self._dropOut, training=self.training)