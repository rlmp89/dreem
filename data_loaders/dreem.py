from base import BaseDataLoader
import torch
import numpy as np
import glob, os, h5py
import pandas as pd
from torch.utils import data
from utils.util import band_filter
from functools import partial
import sys
this = sys.modules[__name__]

class DreemDataset(data.Dataset):
    def __init__(self, file_path,transform=[],training=True,testing=False):
        self.fpath = [f for f in glob.glob(os.path.join(file_path, '*.h5'))][0]
        assert self.fpath, "No file could be loaded"
        self.training = training
        self.testing = testing
        self.data = h5py.File(self.fpath,'r').get('features') #ballec on load en rame

        self.transform = {t['type']:getattr(this,t['type'])(**t['args']) for t in transform }

        if "outliers" in self.transform.keys() and self.training:
            outliers = self.transform['outliers']
            self.outliers=[]
            for o in outliers:
                for k in range(self.data.shape[1]):
                    self.outliers.append(o + self.data.shape[0]*k)
                    self.outliers = sorted(self.outliers)
        else:
            self.outliers = []

        N = self.data.shape[0]
        keep_idx = np.delete(np.arange(N), self.outliers)
        self.data = np.vstack([k.squeeze(axis=1) for k in np.split(self.data[keep_idx,:,:],40,axis=1)])
   

        if self.training or self.testing:
          labelspath =  [f for f in glob.glob(os.path.join(file_path, '*.csv'))][0]
          self.labels = pd.read_csv(labelspath,index_col='id').values
          self.labels = np.vstack([self.labels[keep_idx] for _ in range(40)])
      

        
            

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        # get data
        x = torch.from_numpy(self.data[idx, :, :])

        for k,v in self.transform.items():
            if k!='outliers':
                x= v(x)
        
      
        # get label
        if self.training or self.testing:
          y = torch.from_numpy(self.labels[idx]).squeeze()
          return (x ,y)
        else:
          return x
    

class DreemDataLoader(BaseDataLoader):
    """
    Sleep data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, testing=False, transform={}):
        self.data_dir = data_dir
        self.dataset = DreemDataset(self.data_dir, transform=transform, training=training, testing=testing)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)





from torchaudio.transforms import Spectrogram
class spectrogram(object):
    """Apply spectrogram
    Args: 
        fft: n sample
    """
    def __init__(self, nfft):
        self.spectro = Spectrogram(nfft,normalized=True,power=2)
    def __call__(self, sample):
        return  self.spectro(sample)
    def __name__(self):
        return "spectrogram"


def outliers(out):
    return out
