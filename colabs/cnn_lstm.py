# -*- coding: utf-8 -*-
"""CNN_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10sjiIjOyIJm_nSTvadez3vjUs1pE64qz
"""

import time
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional
import tensorboard
import tensorflow
import datetime
from time import time
import cv2
import copy
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils import data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import PIL
import pandas as pd
import glob
from torchvision.transforms import functional
from torchvision.io import read_image
from torchvision.io.image import read_file
import zipfile as z
from zipfile import ZipFile
from PIL import Image
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from tensorboard import notebook
#%load_ext tensorboard

"""# New Section"""

# hyperparameters
epochs = 30       
batch_size = 40  
learning_rate = 1e-3
display_interval = 10
begin, end, skip = 1, 26, 2
num_classes = 101 
#CNN hyperparameters
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768 # designer option
CNN_embed_dimin = 512   # latent dim extracted by 2D CNN
dropout = 0.25       # dropout probability

# LSTM hyperparameters
RNN_hidden_layers = 2
RNN_hidden_nodes = 512
RNN_FC_dim = 256
display_interval = 10
 # interval for displaying training info



class Dataset(data.Dataset):
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
    
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)
  

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)     
        y = torch.LongTensor([self.labels[index]])                  

      #  print(X.shape)
        return X, y

#!wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
#!wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
#!wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
#!cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
#!unzip ucf101_jpegs_256.zip
''''download preprocessed dataset in parts, concatenate and unzip into a single folder each 
folder '''    
# set paths
data_path = '/content/jpegs_256' # data path  
action_name_path = '/content/UCF101actions.pkl' # Actions_folder 
save_model_path = "/content/drive/MyDrive/results"   
params_image = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if  torch.cuda.is_available()  else {}

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)
#converting the labels to categories
le = LabelEncoder()
le.fit(action_names)


actions = []
fnames = os.listdir(data_path)
 # collecting video names and corresponding labels    
all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g') 
    actions.append(f[(loc1+2): loc2])

    all_names.append(f)
# convert labels to characters 
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)
# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)
transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
selected_frames = np.arange(begin, end, skip).tolist() # selected frames amongst frames
# collection in each folder

train_set, test_set = Dataset(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params_image)
test_loader = data.DataLoader(test_set, **params_image)

class CNN_pretrained(nn.Module):
    def __init__(self, fc_hidden1, fc_hidden2, CNN_embed_dim):
        
        super(CNN_pretrained, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2 #

        resnet = models.resnet152(pretrained=True)# get the resnet model
       # delete the last fc layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) # get the model with the last FC layer
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1) # the last FC is 2048 by 1000, thus , the resnet.fc_features is
         # the output of the layer before the last FC.
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim) #CNN_
        
    def forward(self, frames_):
        cnn_embed_seq = []
        for t in range(frames_.size(1)): #x_3d is in batchsize, sequence/the number of frames and dimensions
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(frames_[:,t, :, :, :])  # ResNet input is  
               # [the batch number, the sequence num, the 3 channel, the H_dim, W_dim ]
               # as per the normal inputs to the architecture
               #output is the batch size 
                   
                x = x.view(x.size(0), -1) 
                       # flatten output of conv with the batch as the pivot of 
                       # the flattening thus you have a flattened tensor with the
                       # batches for each frame
                #the output is a flattened tensor form a tensor of [batch,2048,1,1]
                # to [batch,2048]
            # FC layers
                #print(x.shape)
                x = self.bn1(self.fc1(x))
               # print(x.shape)
            #batch normalise individual tensors of [batch,1024]
                x = F.relu(x)
                x = self.bn2(self.fc2(x))
                x = F.relu(x)
                x = self.fc3(x)
           #the cnn out for 1 sequence, is the [batch,512]
                cnn_embed_seq.append(x)## output x is a tuple of vectors
        # since the tuple, [batch,512] is stacked temporally, we get a time 
        # dependent sequence of batches and flattened feature extracted info
        # swap time and batch size (batch, time dim, dim from CNN) because in 
        # the LSTM setup the batch_first option is chosen , thus it is not seq
        # first
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, frames, input_size)

        return cnn_embed_seq

#resnet = models.inception_v3(pretrained=True)# get the resnet model
#modules = list(resnet.children())[:-1]  
#print(modules)
print(resnet.fc.in_features)

class RNN(nn.Module):
    def __init__(self, CNN_embed_dim, h_RNN_layers, h_RNN, h_FC_dim, drop_p, num_actions):
        super(RNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        # the dimension of [batch,sequence len , input_size]
        # each LSTM node takes [batch, input]
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        # hiddenlayers the LSTM
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        # fully connected stacked layers on the LSTM
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)#Dense layer of output fro LSTM to 512
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)# then another from 512
        # to the number of classes.

    def forward(self, Input):
        
        self.LSTM.flatten_parameters()
        LSTM_output, (hidden_state, cell_state) = self.LSTM(Input, None)  
        # hidden_state size= [LSTM_layers, batchsize, hidden_size),
        # (n_layers, batchsize, hidden_size) 
        #  LSTM_output=[batch, time_step, output_size] 
        
        # FC layers
        x = self.fc1(LSTM_output[:, -1, :])   # choose RNN_out at the last time step
        """ Thus we select the last time step which has thbe time history encapsulated in the hidden state"""
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=True)
        x = self.fc2(x)

        return x



#from google.colab import files
#uploaded = files.upload()
#import io
# get paths of images
#paths_healthy_brain = glob.glob('/content/ucf101_jpeg')
#paths_tumor = glob.glob('/content/UCF101actions.pkl')

#####################################################
epoch_train_losses = []
epoch_train_scores  = []
epoch_test_losses = []
epoch_test_scores = []
###################################################

###############################################################################################################



######################################data #################################################





###################################################################################################

def train(display_interval, model, device, train_loader, optimizer, epoch):
    cnn, rnn = model
    cnn.train()
    rnn.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0) # the number of samples

        optimizer.zero_grad()
        output = rnn(cnn(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch+ 1) % display_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores
    ##############################################################################################
def test(model, device, optimizer, test_loader):
        # set model as testing mode
        cnn, rnn = model
        cnn.eval()
        rnn.eval()

        test_loss =  0
        all_y = []
        all_y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
            # distribute data to device
                X, y = X.to(device), y.to(device).view(-1, )

                output = rnn(cnn(X))

                loss = F.cross_entropy(output, y, reduction='sum')
                test_loss += loss.item()                 # sum up batch loss
                y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
                all_y.extend(y)
                all_y_pred.extend(y_pred)

        test_loss /= len(test_loader.dataset)

    # compute accuracy
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0)
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
        torch.save(cnn.state_dict(), os.path.join(save_model_path, 'cnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        torch.save(rnn.state_dict(), os.path.join(save_model_path, 'rnn_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

        return test_loss, test_score
    
#################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################################################


######################################################################################################
encoder = CNN_pretrained(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                          CNN_embed_dim=CNN_embed_dimin).to(device)
decoder = RNN(CNN_embed_dim=CNN_embed_dimin, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout, num_actions=num_classes).to(device)
##########################################################################################################                         
nn_params = list(encoder.fc1.parameters()) + list(encoder.bn1.parameters()) + \
                  list(encoder.fc2.parameters()) + list(encoder.bn2.parameters()) + \
                  list(encoder.fc3.parameters()) + list(decoder.parameters()) # consider the parameters of 
                  #those added layers
##########################################################################################################
optimizer = torch.optim.Adam(nn_params, lr=learning_rate)
##########################################################################################################                  
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(display_interval, [encoder, decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = test([encoder, decoder], device, optimizer, test_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)



#
 #   writer.add_scalar('Loss/train', float(loss), epoch)
 #   writer.add_scalar('Loss/validate', float(loss), epoch)
 #   writer.add_scalar('Accuracy/train', float(loss), epoch)
  #  writer.add_scalar('Accuracy/validate', float(loss), epoch)

fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epoch + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epoch + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epoch + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epoch+ 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_ResNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()

plt.plot(np.arange(1, epoch + 1), A[:, -1])  # train loss (on epoch end)