from utils import loadVideoFrames, explode
from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        h1, h2, h3 = 32, 64, 128
        self.conv1 = nn.Conv2d(3, h1, 3, padding=1)
        self.conv2 = nn.Conv2d(h1, h2, 3, padding=1)
        self.conv3 = nn.Conv2d(h2, h3, 3, padding=1)
        self.pool = nn.MaxPool2d(4)
        self.dropOut = nn.Dropout3d(self._dropOut)

        # 16711680
        self.flatternDimensions = int(
            h3 * int(self._imageWidth / 4 / 4 / 4) * int(self._imageHeight / 4 / 4 / 4)
        )

        self.fc1 = nn.Linear(self.flatternDimensions, 2048)
        self.lstm = nn.LSTM(2048, 1024, num_layers=self._features, batch_first=False)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 768)
        self.fc4 = nn.Linear(768, self._features)

    def forward(self, x, y):
        self.hidden = None
        # I'm adding a 0 dimensions the lenght of the sequence. My sequence is allways 1
        batches = (
            loadVideoFrames(self._config, x).to(self._device).permute(1, 0, 2, 3, 4)
        )
        y = explode(y, self._minFrames)

        convertedBatch = torch.zeros(
            [0, batches.shape[1], self.flatternDimensions],
            dtype=torch.float32,
            device=self._device,
        )

        for batch in batches:
            f = F.relu(self.pool(self.conv1(batch)))
            f = F.relu(self.pool(self.conv2(f)))
            f = self.dropOut(F.relu(self.pool(self.conv3(f))))
            f = f.flatten(1).unsqueeze(0)
            convertedBatch = torch.cat([convertedBatch, f])

        x = F.relu(self.fc1(convertedBatch))
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropOut(F.relu(self.fc2(x.permute(1, 0, 2))))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2]), y.reshape(
            y.shape[0] * y.shape[1]
        )