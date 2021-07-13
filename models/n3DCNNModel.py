from utils import loadVideoFrames
from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class n3DCNNModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d((2, 4, 4))
        self.conv3 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout3d(0.4)
        self.dropoutl = nn.Dropout(0.6)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(261120, 1024)
        self.fc3 = nn.Linear(1024, 60)
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x, y):
        batches = loadVideoFrames(self._config, x)
        batches = batches.to(self._device)

        x = batches.permute(0, 2, 1, 3, 4)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))

        x = self.flat(x)
        x = self.dropoutl(x)
        x = F.relu(self.fc1(x))

        x = self.fc3(x)

        return x, y
