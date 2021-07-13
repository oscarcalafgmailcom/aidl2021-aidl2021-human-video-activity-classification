from models.BaseModel import BaseModel
from utils import loadVideoFrames
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16LSTMModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        self.vgg16 = models.vgg16(pretrained=True, progress=True)
        self.vgg16.classifier = nn.Identity()
        self.fc1 = nn.Linear(25088, 4096)
        self.lstm = nn.LSTM(4096, 4096, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, self._features)
        self.norm = nn.BatchNorm1d(1000)

    def forward(self, x, y):
        # I'm adding a 0 dimensions the lenght of the sequence. My sequence is allways 1
        self.hidden = None
        batches = loadVideoFrames(self._config, x).to(self._device)
        convertedBatch = torch.zeros(
            [0, batches.shape[0], 25088], dtype=torch.float32, device=self._device
        )

        # exploding x
        for batch in batches.permute(1, 0, 2, 3, 4):
            with torch.no_grad():
                x = self.vgg16(batch)
            convertedBatch = torch.cat((convertedBatch, x.unsqueeze(0)))

        # exploding y
        convy = torch.zeros(
            0, self._minFrames, dtype=torch.long, device=self._device
        )
        for m in range(y.shape[0]):
            convyy = torch.zeros(
                1, self._minFrames, dtype=torch.long, device=self._device
            )
            convyy.fill_(y[m])
            convy = torch.cat((convy, convyy))

        convertedBatch = convertedBatch.permute(1, 0, 2)
        x = F.relu(self.fc1(convertedBatch))
        del batches, convertedBatch

        x, self.hidden = self.lstm(x, self.hidden)
        # x = torch.mean(x, 1, True) it's the diference between this and LSTMModel
        x = x.squeeze()
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x), convy
