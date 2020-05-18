import os
import io
import numpy as np
from PIL import Image
import PIL
import torch

#from scipy.misc.pilutil import imread 


class Bowl2Balls(object):
    
    """Data Handler that loads 2 balls in a Bowl synthetic data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64, epoch_size=300, **kwargs):
        self.root_dir = data_root 
        if train:
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/test' % self.root_dir
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            self.dirs.append('%s/%s/render' % (self.data_dir, d1))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0
        self.N = epoch_size
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def get_seq(self, index):
        d = self.dirs[index%(len(self.dirs))]
        image_seq = []
        l_seq = len([f for f in os.listdir(d) if f[-4:]=='.jpg'])
        start = np.random.randint(0,l_seq-self.seq_len-1)
        for i in range(start, start+self.seq_len):
            fname = '%s/%04d.jpg' % (d, 3*i)
            im = (np.asarray(Image.open(fname).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, self.image_size, self.image_size, 3).astype('float32') - 127.5) / 255
            image_seq.append(im)
        image_seq = np.concatenate(image_seq, axis=0)
        if self.seq_len == 1:
            image_seq = torch.from_numpy(image_seq[0,:,:,:]).permute(2,0,1)
        else:
            image_seq = torch.from_numpy(image_seq[:,:,:,:]).permute(0,3,1,2)
        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        # return {'images': self.get_seq(index)}
        return self.get_seq(index)

