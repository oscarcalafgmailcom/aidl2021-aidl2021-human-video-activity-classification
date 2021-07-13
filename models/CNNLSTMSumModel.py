from utils import loadVideoFrames, explode
from models.CNNLSTMModel import CNNLSTMModel
from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMSumModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        h1, h2, h3 = 32, 128, 256
        d1, self.d2, d3, d4, d5 = 512, 256, 256, 128, 128
        pooling = 4
        self.conv1 = nn.Conv2d(3, h1, 3, padding=1)
        self.convbn1 = nn.BatchNorm2d(h1, momentum=0.01)
        self.conv2 = nn.Conv2d(h1, h2, 3, padding=1)
        self.convbn2 = nn.BatchNorm2d(h2, momentum=0.01)
        self.conv3 = nn.Conv2d(h2, h3, 3, padding=1)
        self.convbn3 = nn.BatchNorm2d(h3, momentum=0.01)
        self.pool = nn.MaxPool2d(pooling)

        # 16711680
        self.flatternDimensions = int(
            h3
            * int(self._imageWidth / pooling / pooling / pooling)
            * int(self._imageHeight / pooling / pooling / pooling)
        )
        self.prefc1 = nn.Linear(self.flatternDimensions, d1)
        self.prebn1 = nn.BatchNorm1d(d1, momentum=0.01)
        self.prefc2 = nn.Linear(d1, self.d2)
        self.prebn2 = nn.BatchNorm1d(self.d2, momentum=0.01)

        self.lstm = nn.LSTM(self.d2, d3, num_layers=3, batch_first=False)
        self.bnlstm = nn.BatchNorm1d(d3, momentum=0.01)
        self.fc1 = nn.Linear(d3, d4)
        self.bn = nn.BatchNorm1d(d4, momentum=0.01)
        self.fc2 = nn.Linear(d4, d5)
        self.fc3 = nn.Linear(d5, self._features)

    def forward(self, x, y):
        self.hidden = None
        # I'm adding a 0 dimensions the lenght of the sequence. My sequence is allways 1
        batches = loadVideoFrames(self._config, x).to(self._device)
        batches = batches.permute(1, 0, 2, 3, 4)
        dims = [0, batches.shape[1], self.d2]
        enc = torch.zeros(dims, dtype=torch.float32, device=self._device)

        for batch in batches:
            f = self.convbn1(F.relu(self.pool(self.conv1(batch))))
            f = self.convbn2(F.relu(self.pool(self.conv2(f))))
            f = self.convbn3(F.relu(self.pool(self.conv3(f))))
            f = self.prebn1(F.relu(self.prefc1(f.flatten(1))))
            f = self.dropout(self.prebn2(F.relu(self.prefc2(f))))
            enc = torch.cat((enc, f.unsqueeze(0)))

        x, (self.hidden, h_n) = self.lstm(enc, self.hidden)
        x = self.dropout(self.bnlstm(F.relu(x[-1, :, :])))
        x = self.dropout(self.bn(F.relu(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        del enc, f, batches, dims
        return x, y
