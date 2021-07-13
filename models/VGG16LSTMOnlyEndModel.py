from utils import loadVideoFrames
from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16LSTMOnlyEndModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        self.vgg16 = models.vgg16(pretrained=True, progress=True)
        self.vgg16.classifier = nn.Identity()
        d1, self.d2, d3, d4, d5 = 25088, 1000, 512, 128, 128
        self.prefc1 = nn.Linear(d1, self.d2)
        self.prebatchnorm = nn.BatchNorm1d(self.d2)
        self.lstm = nn.LSTM(self.d2, d3, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(d3, d4)
        self.batchnorm = nn.BatchNorm1d(d4)
        self.fc3 = nn.Linear(d4, self._features)

    def forward(self, x, y):
        # I'm adding a 0 dimensions the lenght of the sequence. My sequence is allways 1
        self.hidden = None
        batches = loadVideoFrames(self._config, x).to(self._device)
        convertedBatch = torch.zeros(
            [0, batches.shape[0], self.d2], dtype=torch.float32, device=self._device
        )

        for batch in batches.permute(1, 0, 2, 3, 4):
            with torch.no_grad():
                x = self.vgg16(batch)
            x = self.dropout(self.prebatchnorm(F.relu(self.prefc1(x))))
            convertedBatch = torch.cat([convertedBatch, x.unsqueeze(0)])

        convertedBatch = convertedBatch.permute(1, 0, 2)
        x, (self.hidden, c_n) = self.lstm(convertedBatch, self.hidden)
        x = self.dropout(self.batchnorm(F.relu(self.fc2(x[:, -1, :]))))
        return self.fc3(x), y
