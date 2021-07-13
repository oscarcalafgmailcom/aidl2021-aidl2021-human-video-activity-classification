from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNLSTM(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        self.fcconv = nn.Linear(14336, self.input_size)
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.fc1 = nn.Linear(5120, 2048)
        self.fc2 = nn.Linear(2048, 60)
        # self.fc3=nn.Linear(512,60)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=512, kernel_size=3, padding=1
        )
        # self.conv4=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.features = torch.empty((self._batchSize, 0, 512, 4, 7))
        self.batch_size = self._batchSize

    def forward(self, x, y):
        # I'm adding a 0 dimensions the lenght of the sequence. My sequence is allways 1
        features = self.features.to(self._device)
        for frame in x:
            x = F.relu(self.pool(self.conv1(frame)))
            x = F.relu(self.pool(self.conv2(x)))
            x = F.relu(self.pool(self.conv3(x)))
            # x=F.relu(self.pool(self.fc4(x)))
            x = torch.reshape(x, (self.batch_size, 1, 512, 4, 7))
            features = torch.cat((features, x), dim=1)
        features = torch.flatten(features, start_dim=2, end_dim=-1)
        x = F.relu(self.fcconv(features))
        x, (hn, cn) = self.lstm(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, y
