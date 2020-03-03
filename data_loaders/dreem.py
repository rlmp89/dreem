from base import BaseDataLoader
import torch
import numpy as np
import glob, os, h5py
import pandas as pd
from torch.utils import data
from utils.util import band_filter
from functools import partial

class DreemDataset(data.Dataset):
    def __init__(self, file_path,transform={},training=True,testing=False):
        self.fpath = [f for f in glob.glob(os.path.join(file_path, '*.h5'))][0]
        assert self.fpath, "No file could be loaded"
        self.training = training
        self.testing = testing
        self.data = h5py.File(self.fpath,'r').get('features') #ballec on load en ram
        self.data_shape = self.data.shape
        self.data=np.vstack([k.squeeze(axis=1) for k in np.split(self.data,40,axis=1)])
        if self.training or self.testing:
          labelspath =  [f for f in glob.glob(os.path.join(file_path, '*.csv'))][0]
          self.labels = pd.read_csv(labelspath,index_col='id').values
        self.transform = transform

        if "outliers" in self.transform.keys() and self.training:
            outliers = self.transform['outliers']
            self.outliers=[]
            
            for o in outliers:
                for k in range(self.data_shape[1]):
                    self.outliers.append(o + self.data_shape[0]*k)
                    self.outliers = sorted(self.outliers)
        else:
            self.outliers = []

        N = self.data_shape[0]*self.data_shape[1]
        self.index_map = np.delete(np.arange(N), self.outliers)
        
            

    def __len__(self):
            return len(self.index_map)

    def __getitem__(self, idx):
        # get data
        # access to h5py inside __getitem__ mandatory in order to be able to use multiprocessing
        n_subjects = self.data_shape[0]
        IDX =  self.index_map[idx]
        x = self.data[IDX%n_subjects, :, :]

        '''if "band_filter" in self.transform:
            low,high = self.transform['band_filter']
            x= np.apply_along_axis(partial(band_filter,low=low,high=high),axis=1,arr=x)'''
        x = torch.from_numpy(x).float()
      
        # get label
        if self.training or self.testing:
          y = self.labels[IDX%n_subjects]
          y = torch.from_numpy(y).squeeze()
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
