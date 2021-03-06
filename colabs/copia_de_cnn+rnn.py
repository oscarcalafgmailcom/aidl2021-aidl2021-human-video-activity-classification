# -*- coding: utf-8 -*-
"""Copia de cnn+rnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NIg-5S35CtvGsXgL0qA03ck-uvte6DJi
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torchvision.transforms as transforms
from torchvision.transforms import functional
from torchvision.io import read_image
from torchvision.io.image import read_file
from torch import torch
import cv2
from sklearn import preprocessing
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import torchvision
import torch.nn as nn
from torch import optim as optim

class NTUDataset(Dataset):
    def __init__(self, criteria, config):
        self.criteria = criteria
        self.files = glob.glob(self.criteria)
        #self.load_as_images = load_as_images
        self.transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((272,480))
                                          ,transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.imageResize = config["imageResize"]
        self.imageResizeWidth = config["imageResizeWidth"]
        self.imageResizeHeight = config["imageResizeHeight"]
        self.seq_len = config["seq_len"]
        self.labels = []
        for f in self.files:
            self.labels.append(f[len(f)-12:-8])

        self.labels.sort()
        self.labels = list(set(self.labels))
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = path[len(path)-12:-8]

        value = self.labels.index(label)

        
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0
        fcount =0
        
        
        seq=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                
        seq=self.transform(seq)
        seq= torch.unsqueeze(seq,0)
        while success:
            fcount+=1
            if (fcount) % 10 == 0:
                
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                #trans1=transforms.ToPILImage()
                 #trans2=transforms.ToTensor()
                image=self.transform(image)
                #plt.imshow(image)
                image=torch.unsqueeze(image,0)
           
                #cv2.imwrite('../savedimage.jpg',image)
            
                seq = torch.cat((seq,image),0)
            
                count+=1
                if seq.shape[0] == 10:
                    break
            success,image=vidcap.read()
            vidcap.release
        
        lab= value
        lab= torch.tensor(lab,dtype= torch.long)
        
        if seq.shape[0] < 10:
            pad=torch.zeros(10-seq.shape[0],3,272,480)
            seq=torch.cat((seq,pad),0)

        #seq=seq.permute(1,0,2,3)#Solo para el caso de Conv3d    

        return seq, lab,count

config = {
        "lr": 0.0001,
        "batch_size": 4,
        "epochs": 20,
        "h1": 64,
        "h2": 128,
        "h3": 256,
        "h4": 4096,
        "h5": 1024,
        "features": 60,
        "imageInputWidth": 1920,
        "imageInputHeight": 1080,
        "imageResize": True,
        "imageResizeWidth": 480,
        "imageResizeHeight": 272,
        "seq_len":120,
        "hidden_size": 512,
        "num_layers": 1,
        "input_size":1000,
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splitDataset(dataset: Dataset, train=90, test=5, val=5):
  len = dataset.__len__()
  propsum = train+test+val
  longtrain = int(train * len / propsum)
  longtest = int(test * len / propsum)
  longval = int(len - longtrain - longtest)
  return torch.utils.data.random_split(dataset, [longtrain, longtest, longval])

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')
dataset = NTUDataset("/content/drive/MyDrive/nturgb+d_rgb/*.avi", config)

train_dataset, val_dataset, test_dataset = splitDataset(dataset, 50, 10, 40)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().cpu().numpy().mean()
    return acc

#modelvgg=torchvision.models.vgg16(pretrained=True)
 #modelvgg=torch.nn.Sequential(*list(modelvgg.features.children())[:-1])
 #for param in modelvgg.parameters():
  # param.requires_grad=False

class MyModel(nn.Module):

    def __init__(self,input_size=1000, hidden_size=256,num_layers=2,batch_size=4):
      super().__init__()
      self.input_size=input_size
      self.num_layers= num_layers
      self.hidden_size= hidden_size

      self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
      self.conv2=nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3,padding=1)
      self.conv3=nn.Conv2d(in_channels=128,out_channels=512,kernel_size=3,padding=1)
      self.pool=nn.MaxPool2d(4)

      self.fcconv= nn.Linear(14336,self.input_size)
      self.lstm=nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
      self.fc1 = nn.Linear(5120, 2048)
      self.fc2= nn.Linear(2048,60)
      #self.fc3=nn.Linear(512,60)
      #self.conv4=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
      self.features=torch.empty((batch_size,0,512,4,7))
      self.batch_size=batch_size
      self.drop=nn.Dropout(0.5)        

    def forward(self, x):
      features=self.features.to(device)
      for frame in x:
        x=F.relu(self.pool(self.conv1(frame)))
        x=F.relu(self.pool(self.conv2(x)))
        x=F.relu(self.pool(self.conv3(x)))
        x=torch.reshape(x,(self.batch_size,1,512,4,7))
        features=torch.cat((features,x),dim=1)

      features=torch.flatten(features,start_dim=2,end_dim=-1)
      y = self.drop(features)
      y= F.relu(self.fcconv(y))
      y,(hn,cn) = self.lstm(y)
      y=torch.flatten(y, start_dim=1,end_dim=-1)
      y=F.relu(self.fc1(y))
      y=self.fc2(y)
      
        

      return y

class Modelvgg16(nn.Module):

    def __init__(self,input_size=1000, hidden_size=256,num_layers=2):
        super().__init__()
        self.input_size=input_size
        self.num_layers= num_layers
        self.hidden_size= hidden_size
        self.vgg=torchvision.models.vgg16(pretrained=True)
        self.fcconv= nn.Linear(261120,self.input_size)

        self.lstm=nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        self.fc1 = nn.Linear(10240, 2048)
        self.fc2= nn.Linear(2048,512)
        self.fc3=nn.Linear(512,60)

        

    def forward(self, x):
        x= F.dropout(F.relu(self.fcconv(x)), p=0.3, training=self.training)
        x,(hn,cn) = self.lstm(x)
        x=torch.flatten(x, start_dim=1,end_dim=-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

def train(config, model, train_loader, optimizer):
    model.train()
    accs, losses = [], []
    lfunct=nn.CrossEntropyLoss()
    
    for x, y,count in train_loader:
        
        x, y = x.to(device), y.to(device)
        x= x.permute (1,0,2,3,4)
        optimizer.zero_grad()
        
        hn = model(x)
        loss = lfunct(hn, y)
        loss.backward()
        # Optimitzador
        optimizer.step()
        acc = accuracy(y, hn)
        # Ho posem al vector de losses i accuracies
        losses.append(loss.item())
        accs.append(acc.item())
        #print(loss,acc)

    return np.mean(losses), np.mean(accs)

def test(model, val_loader):
  accs, losses = [], []
  lfunct=nn.CrossEntropyLoss()
  with torch.no_grad():
    model.eval()
    for x, y,count in val_loader:
      x, y = x.to(device), y.to(device)
      x= x.permute (1,0,2,3,4)
      hn = model(x)
      loss = lfunct(hn, y)
      acc = accuracy(y, hn)
      losses.append(loss.item())
      accs.append(acc.item())
      
    return np.mean(losses), np.mean(accs)

model = MyModel(config["input_size"],config["hidden_size"],config["num_layers"],config['batch_size']).to(device)
 train_acc=[]
 test_acc=[]
 train_loss=[]
 test_loss=[]
 root="/content/drive/MyDrive/results/checkpointcr"
 optimizer = optim.Adam(model.parameters(), config["lr"],weight_decay=0.0001)
 for epoch in range(config["epochs"]):
   loss, acc = train(config, model, train_loader, optimizer)
   losst, acct = test(model, val_loader)
   train_acc.append(acc)
   test_acc.append(acct)
   train_loss.append(loss)
   test_loss.append(losst)


   print(f"Epoch {epoch} loss={loss:.12f} acc={acc:.12f} lossT={losst:.12f} accT={acct:.12f}")
   checkpointFilename= f"{root}{epoch}.pt"

   checkpoint={
    "modelstatedict":model.state_dict(),
    "optstatedict":optimizer.state_dict(),
    "trainlosses": train_loss,
    "testlosses": test_loss,
    "trainacc": train_acc,
    "testacc":test_acc,
    "init_epoch2": epoch,
      }

   torch.save(checkpoint,checkpointFilename)

model = MyModel(config["input_size"],config["hidden_size"],config["num_layers"],config['batch_size']).to(device)
 root="/content/drive/MyDrive/results/checkpointcr8"
 optimizer = optim.Adam(model.parameters(), config["lr"],weight_decay=0.0001)
 checkpointfname=f"{root}.pt"
 checkpoint=torch.load(checkpointfname,map_location="cuda:0")
 train_acc=checkpoint["trainacc"]
 test_acc=checkpoint["testacc"]
 train_loss=checkpoint["trainlosses"]
 test_loss=checkpoint["testlosses"]
 initepoch=checkpoint["init_epoch2"]
 model.load_state_dict(checkpoint["modelstatedict"])
 optimizer.load_state_dict(checkpoint["optstatedict"])
 model.to(device)
 for epoch in range(initepoch+1,config["epochs"]):
   loss, acc = train(config, model, train_loader, optimizer)
   losst, acct = test(model, val_loader)
   train_acc.append(acc)
   test_acc.append(acct)
   train_loss.append(loss)
   test_loss.append(losst)


   print(f"Epoch {epoch} loss={loss:.12f} acc={acc:.12f} lossT={losst:.12f} accT={acct:.12f}")
   checkpointFilename= f"{root}{epoch}.pt"

   checkpoint={
    "modelstatedict":model.state_dict(),
    "optstatedict":optimizer.state_dict(),
    "trainlosses": train_loss,
    "testlosses": test_loss,
    "trainacc": train_acc,
    "testacc":test_acc,
    "init_epoch2": epoch,
      }

   torch.save(checkpoint,checkpointFilename)