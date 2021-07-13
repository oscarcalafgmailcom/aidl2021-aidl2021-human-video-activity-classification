from models.BaseModel import BaseModel
from utils import loadVideoFrames
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet101


class RESNET101LSTMModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config=config, device=device)
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self._features)

    def forward(self, x, y):

        batches = loadVideoFrames(self._config, x).to(self._device)
        hidden = None
        for t in range(batches.size(1)): # 50, 20, 3, 480, 172
            with torch.no_grad():
                x = self.resnet(batches[:, t, :, :, :]) # 50, 3, 48, 172 but the position t 
            out, hidden = self.lstm(x.unsqueeze(0), hidden) # 1, 50, 300

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x, y
