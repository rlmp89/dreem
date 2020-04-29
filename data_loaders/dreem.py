from base import BaseDataLoader
import torch
import numpy as np
import glob, os, h5py
import pandas as pd
from torch.utils import data    
from functools import partial
from collections import OrderedDict
import sys

from . import transformers

class DreemDataset(data.Dataset):
    def __init__(self, file_path,transform=[],training=True,testing=False):
        self.fpath = [f for f in glob.glob(os.path.join(file_path, '*.h5'))][0]
        assert self.fpath, "No file could be loaded"
        self.training = training                
        self.testing = testing
        self.data = h5py.File(self.fpath,'r').get('features') # <!>  laoded in ram

        N_idv = self.data.shape[0]
        N_trials = self.data.shape[1]

        #load preprocessing transformers
        self.transform = OrderedDict({t['type']: getattr(transformers,t['type'])(**t.get('args',{})) for t in transform })
        if self.transforms:
          print("Pre-processing pipeline:\n\t- " + "\n\t- ".join([p.__name__() for p in self.transform.values()]))
        
        #remove outliers
        outliers_transf = self.transform.get('outliers')
        self.outliers = outliers_transf(n_idv=N_idv, training=self.training ) if outliers_transf else []
        keep_idx = np.delete(np.arange(N_idv * N_trials), self.outliers )
        self.data = np.vstack([k.squeeze(axis=1) for k in np.split(self.data, N_trials, axis=1)])[keep_idx]
   
        #load labels
        if self.training or self.testing:
          labelspath =  [f for f in glob.glob(os.path.join(file_path, '*.csv'))][0]
          self.labels = pd.read_csv(labelspath,index_col='id').values
          self.labels = np.vstack([self.labels  for _ in range(N_trials)])[keep_idx]
       
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
    Dreem data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, testing=False, transform={}):
        self.data_dir = data_dir
        self.dataset = DreemDataset(self.data_dir, transform=transform, training=training, testing=testing)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)