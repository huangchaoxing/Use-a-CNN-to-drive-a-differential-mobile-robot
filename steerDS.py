import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from os import path
from PIL import Image
import random

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)
        kernel_size = random.randint(1,4)
        img=cv2.blur(img,(kernel_size,kernel_size))
        h = img.shape[0]
        img = img[int(0.6*h):h,:]
        img = Image.fromarray(img)
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(steering)

        steer_int = np.float32([0,1,0])
        if (steering <= -0.10):
            steer_int = np.float32([1,0,0])
        elif (steering >= 0.10):
            steer_int = np.float32([0,0,1])
    
        sample = {"image":img , "steering":steer_int}        
        
        return sample


def testDS():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("../dev_data/training_data",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]
        
        print(im.shape)
        print(y)
        break



if __name__ == "__main__":
    test()