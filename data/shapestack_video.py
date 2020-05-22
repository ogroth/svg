import os
import io
import numpy as np
from PIL import Image
#from scipy.misc.pilutil import imread 
from pathlib import Path
import PIL
import torch

class ShapeStackVideo(object):
    
    """Data Handler that loads ShapeStack data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64, **kwargs):
        self.root_dir = data_root

        self.data_dir = []
        # for val  in ('shapestacks-vcom', 'shapestacks-vpsf'):
        for val in ('shapestacks-vcom', ):
            self.data_dir.append('%s/%s/recordings/' % (self.root_dir, val))
        
        
        # if train:
        #     self.data_split = '%s/splits/default/train.txt' % self.root_dir
        #     self.ordered = False
        # else:
        #     self.data_split = '%s/splits/default/test.txt' % self.root_dir
        #     self.ordered = True 
        # self.dir = []
        # with open(self.data_split) as fp:
        #     for line in fp:
        #         line = line.strip('\n')
        #         self.dir.append('%s%s' % (self.data_dir, line))

        self.dir = []
        for directory in self.data_dir:
            self.dir += [directory+f for f in os.listdir(directory) if f[0]=='e']

        self.dirs = []
        for i in range(len(self.dir)):
            dir_name = self.dir[i]
            if os.path.exists(dir_name) and int(dir_name.split('-h=')[-1][0])<=5:
                name = set([dir_name+'/'+f.split('mono')[0] for f in os.listdir(dir_name) if f[-4:]=='.png'])
                ## Look at the h option and load only 2...
                self.dirs += list(name) 

        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0
        self.N = int(kwargs['epoch_size'])
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def get_seq(self, index):
        d = self.dirs[index%(len(self.dirs))]
        image_seq = []
        # l_seq = len([f for f in os.listdir(d) if f[-4:]=='.png'])
        # name = [f for f in os.listdir(d) if f[-4:]=='.png'][0][:36]
        start = np.random.randint(0,3)
        # for i in range(start, start+7*self.seq_len, 7):
        for i in range(start, start+self.seq_len):
            # fname = '%s/%s%d.png' % (d,name,i)
            # im = (np.asarray(Image.open(d+'mono-%04d.png'%i).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, \
            #     self.image_size, self.image_size, 3).astype('float32') - 127.5) / 255
            im = (np.asarray(Image.open(d+'mono-%04d.png'%i).resize((self.image_size,self.image_size),PIL.Image.LANCZOS)).reshape(1, \
                self.image_size, self.image_size, 3).astype('float32')) / 255
            image_seq.append(im)
        if self.seq_len == 1:
            image_seq = np.concatenate(image_seq, axis=0)
            image_seq = torch.from_numpy(image_seq[0,:,:,:]).permute(2,0,1)
        else:
            image_seq = [torch.from_numpy(im[0,:,:,:]).permute(2,0,1) for im in image_seq]
            image_seq = torch.stack(image_seq,0) 
        return image_seq


    def __getitem__(self, index):
        self.set_seed(index)
        # return {'images': self.get_seq(index)}
        return self.get_seq(index)


