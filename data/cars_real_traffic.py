from torchvision import datasets, transforms
import torch
import os
#from scipy.misc.pilutil import imread 
import numpy as np
from PIL import Image
import PIL

class CarsRealTraffic(object):
    
    """Data Handler that loads cars data."""

    def __init__(self, data_root, train=True, seq_len=1, image_size=64, **kwargs):
        # This is TODO, we want to see what happens here
        self.root_dir = data_root 
        self.image_size = image_size

        self.dir = sorted([self.root_dir+f for f in os.listdir(self.root_dir) if f[0]=='f'])
        # split data in training and test set as 80/20
        split_idx = int(np.rint(len(self.dir) * 0.8))
        if train:
            self.dir = self.dir[:split_idx]  # 80% (491) of 614 videos total
        else:
            self.dir = self.dir[split_idx:]  # 20% (123) of 614 videos total

        self.data = []
        for i in range(len(self.dir)):
            dir_name = self.dir[i]
            seq_ims  = sorted([dir_name+'/'+f for f in os.listdir(dir_name) if f[-5:]=='.jpeg'])
            for j in range(len(seq_ims)-3*seq_len):
                ## Look at the h option and load only 2...
                self.data.append(seq_ims[j:j+2*seq_len:2])
    
        self.N = int(kwargs['epoch_size'])
        self.seq_len = seq_len

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        images = self.data[index%len(self.data)]

        image_seq = []
        for i in range(len(images)):
            # im = (np.asarray(Image.open(images[i]).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, \
            #     self.image_size, self.image_size, 3).astype('float32') - 127.5) / 255
            im = (np.asarray(Image.open(images[i]).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, \
                self.image_size, self.image_size, 3).astype('float32')) / 255
        
            image_seq.append(torch.from_numpy(im[0,:,:,:]).permute(2,0,1))
        
        # return {'images': torch.stack(image_seq,0)}
        return torch.stack(image_seq,0)

