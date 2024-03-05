# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:56:19 2024

@author: burak
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

klasor_yolu = 'img_train/'  
dosya_isimleri = [dosya for dosya in os.listdir(klasor_yolu) if dosya.endswith('.jpg')]

hedef_boyut = (106, 106)

data = []
for dosya in dosya_isimleri:
    tam_yol = os.path.join(klasor_yolu, dosya)
    with Image.open(tam_yol) as img:
        img = img.convert('L').resize(hedef_boyut)  
        numpy_dizi = np.array(img)  
        data.append(numpy_dizi)

x_train = np.array(data)

csv_dosya_yolu = 'train_sol.csv'  
csv_matrisi = pd.read_csv(csv_dosya_yolu).values

y_train = csv_matrisi[:, 1:]
print('Load ok')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 26 * 26, 128)  
        self.fc2 = nn.Linear(128, 37) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 26 * 26)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CustomCNN().to(device)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predicted, actual):
        return torch.sqrt(self.mse(predicted, actual))
    
criterion = RMSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_train_tensor = torch.from_numpy(x_train).float().unsqueeze(1)  

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

batch_size = 32  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
print('transfer ok, traing start')
num_epochs = 100  
loss_values=[]
lr=0.01
for epoch in range(num_epochs): 
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        labels=labels.float()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss = running_loss / len(train_loader)
        loss_values.append(epoch_loss)

        running_loss += loss.item()
        if i % 200 ==0:
                lr=lr*0.995
                optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9)
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.10f}, lr :{lr:.5f}')
            running_loss = 0.0


print('Finished Training')
torch.save(net, 'CNN_SGD.pth')

print('module saved as CNN_SGD.pth')
import pygame
import time

def play_mp3_for_duration(path, duration):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    time.sleep(duration)

    pygame.mixer.music.stop()

mp3_file_path = 'alarm.mp3'
play_mp3_for_duration(mp3_file_path, 2)  
