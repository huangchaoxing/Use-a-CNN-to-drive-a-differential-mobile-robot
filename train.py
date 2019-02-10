# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:47:30 2019

@author: HP
"""

from matplotlib import pyplot as plt
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from data_split import train_valid_split
from steerNet import SteerNet
import torch.optim as optim
from tensorboardX import SummaryWriter
ITER_NUM=60
LR=1e-3
BATCH_SIZE=40
TRAIN_COUNT = 619
VALID_COUNT = 32

train_loader,validation_loader=train_valid_split(batch_size=BATCH_SIZE,valid_portion=0.05)


model = SteerNet()
device = torch.device("cuda")
print(device)
model = model.to(device)
criterion = nn.MSELoss()
# optimizer = optim.Adam(params=model.parameters(),weight_decay=5e-3,lr=LR)

writer = SummaryWriter()


train_loss_dic={}
train_acc_dic = {}
valid_acc_dic = {}
valid_loss_dic={}
print("Ready !")
step=0
for epoch in range(ITER_NUM):
    epoch_loss=0
    epoch_valid_loss=0
    print("This is epoch:",epoch)    
    optimizer = optim.Adam(params=model.parameters(),weight_decay=1e-2,lr=LR)
    # LR = LR*0.95
    right_train = 0
    right_valid = 0
    for data in train_loader:
        image=data["image"].to(device)
        label=data["steering"].to(device)
        optimizer.zero_grad()
        output=model(image)
        output = torch.squeeze(output,1)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        prediction = torch.argmax(output.data,1)
        
        # print("pred",prediction)
        # print("la",torch.argmax(label,1))
        right_train = right_train + (prediction == torch.argmax(label,1)).sum().item()
        # print("eq",(prediction == torch.argmax(label,1)))
    
        step=step+1

    train_acc_dic[epoch] = right_train/TRAIN_COUNT        
    train_loss_dic[epoch]=epoch_loss/(TRAIN_COUNT/BATCH_SIZE)
    print("------------------------")
    print("The epoch acc is,",train_acc_dic[epoch],"The epoch loss is,",train_loss_dic[epoch])


    step = 0
    for data in validation_loader:
        image=data["image"].to(device)
        label=data["steering"].to(device)
        output=model(image)
        output = torch.squeeze(output,1)
        loss=criterion(output,label)
        
        epoch_valid_loss+=loss.item()
        

        prediction = torch.argmax(output.data,1)
        right_valid = right_valid + (prediction == torch.argmax(label,1)).sum().item()
    

        # _, prediction = torch.max(output.data, 1)
        # right_valid = right_valid + (prediction == label).sum().item()
       

    valid_acc_dic[epoch] = right_valid/VALID_COUNT        
    valid_loss_dic[epoch]= epoch_valid_loss/(VALID_COUNT/BATCH_SIZE)
    
    print("The valid acc is,",valid_acc_dic[epoch],"The valid loss is,",valid_loss_dic[epoch])    
    print("******************************")

print("training finished !")


model = model.to(torch.device("cpu"))
torch.save(model.state_dict(), "ourNet.pt")



plt.figure(1)
eps=train_loss_dic.keys()
plt.plot(eps,train_loss_dic.values())    
plt.show()
plt.figure(2)
plt.plot(eps,valid_loss_dic.values())
plt.show()

plt.figure(3)
plt.plot(eps,valid_loss_dic.values(),eps,train_loss_dic.values())
plt.show()

plt.figure(4)
plt.plot(eps,train_acc_dic.values(),eps,valid_acc_dic.values())
plt.show()

