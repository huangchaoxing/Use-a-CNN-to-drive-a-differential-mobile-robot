# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:59:13 2019

@author: HP
"""

import torch
import numpy as np
import torchvision
#from utils import plot_images
import steerDS
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def train_valid_split(batch_size,valid_portion):
    
    normalization=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
    transform= transforms.Compose([transforms.Resize((72,72)),transforms.ToTensor(),normalization])
    
    data_set=steerDS.SteerDataSet("../dev_data/training_data",".jpg",transform)
       
    num_data=len(data_set)
    
   
    indices=list(range(num_data))
    
    split=int(np.floor(valid_portion*num_data))
                                                                
    train_index,valid_index=indices[split:],indices[:split]  #prepare the spilt index for training and validation
    print("Train")
    print(len(train_index))
    print(len(valid_index))
    
    training_sampler=SubsetRandomSampler(train_index)
    validation_sampler=SubsetRandomSampler(valid_index)
    # get the training set and validation set
    
    training_loader=torch.utils.data.DataLoader(data_set,batch_size=batch_size,shuffle=False,sampler=training_sampler,num_workers=0)
    validation_loader=torch.utils.data.DataLoader(data_set,batch_size=batch_size,shuffle=False,sampler=validation_sampler,num_workers=0)
    
    #sample_shower=torch.utils.data.DataLoader(training_set,batch_size=4,shuffle=False,num_workers=0)
    
    #test_loader=torch.utils.data.DataLoader(test_set,num_workers=0)
    
    
    
    return training_loader,validation_loader



# if(__name__ == "__main__"):
#     train_loader,validation_loader=train_valid_split(batch_size=1,valid_portion=0.1)

#     for data in train_loader:
#         image = data["image"]
#         label = data["steering"]
#         imshow(image[0])
#         print(label)


#     # imshow(torchvision.utils.make_grid(images))
#     print(labels)
