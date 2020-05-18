import os
import io
import numpy as np
from PIL import Image
#from scipy.misc.pilutil import imread 
from pathlib import Path
import PIL
import torch

class CLEVRER(object):
    
    """Data Handler that loads ShapeStack data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64, **kwargs):
        self.root_dir = data_root
        self.image_size = image_size

        self.data_dir = [self.root_dir+f for f in os.listdir(self.root_dir) if f[0] == 'v']
        self.dir = []
        for directory in self.data_dir:
            self.dir += [directory+'/'+f for f in os.listdir(directory) if f[0]=='f']
    
        self.N = int(kwargs['epoch_size'])
        self.seq_len = seq_len

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        images = self.dir[index%len(self.dir)]
        seq_ims  = sorted([images+'/'+f for f in os.listdir(images) if f[-5:]=='.jpeg'])
        l_seq = len(seq_ims)
        start = np.random.randint(0,l_seq-self.seq_len-1)
        images = seq_ims[start:start+self.seq_len]
        image_seq = []
        for i in range(len(images)):
            im = (np.asarray(Image.open(images[i]).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, \
                self.image_size, self.image_size, 3).astype('float32') - 127.5) / 255
        
            image_seq.append(torch.from_numpy(im[0,:,:,:]).permute(2,0,1))
        
        # return {'images': torch.stack(image_seq,0).squeeze()}
        return torch.stack(image_seq,0).squeeze()

