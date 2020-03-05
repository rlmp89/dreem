import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from base import BaseModel
import math

def get_out_size(Lin,kernel_sz,stride,padding,dilation=1):
    Lout =  math.floor( (Lin + 2* padding - dilation*(kernel_sz-1) - 1 )/stride +1 )
    return Lout


class DreemModel_nm(BaseModel):#https://arxiv.org/pdf/1703.01789.pdf
    def __init__(self,m, num_classes=2):
            super().__init__()
            self.N_samples = 500
            self.N_chan = 7
            self.m_kernel=m
            #for a  n^m model webuild:
            # a first conv layer with kernel-stride n
            #type   kernel  stride  padding filters output_dim
        # 1 #conv   3       1       0       128     500 x 128
            #------------------------------------------------
        # 2 #conv   3       1       1       128     500 x 128 
            #maxpl  3       3       0       128     166  x 128
            #------------------------------------------------
        # 3 #conv   3       1       1       128     166  x 256 
            #maxpl  3       3       0       256     55  x 256
            #------------------------------------------------
        # 4 #conv   3       1       1       256     55  x 512
            #maxpl  3       3       0       512     18   x 512 
            #------------------------------------------------  
        # 5 #conv   3       1       1       512     18  x 1024
            #maxpl  3       3       0       1024     6   x 1024  

            #1
            self.conv_blocks = [ self.build_block(self.N_chan, self.N_samples, 128, self.m_kernel,1,1,False)]
            k=0
            while self.conv_blocks[-1]['Lout'] > 2*self.m_kernel:
              k+=1
              self.conv_blocks.append(self.build_block( self.conv_blocks[-1]["output_chan"], self.conv_blocks[-1]['Lout'],2**(7+k//2) , self.m_kernel,1,1,True))

            for idx in range(len(self.conv_blocks)):
              for module in ('conv','drop','pool','batch_norm','act'):
                  setattr(self,f"L{idx}_{module}",self.conv_blocks[idx][module])

            in_fc1 =  self.conv_blocks[-1]['Lout'] * self.conv_blocks[-1]['output_chan']
    
            self.fc1 = nn.Linear( in_fc1, 100)
            self.fc2 = nn.Linear(100, num_classes)
            


    def build_block(self,N_input_chan,N_sample,N_output_chan, kernel, stride,padding, max_pool):
        block={}
        block['conv'] = nn.Conv1d(in_channels = N_input_chan, 
                        out_channels = N_output_chan,
                        kernel_size =  kernel,
                        stride =  stride,
                        padding = padding
                        )
        Lout = get_out_size( Lin = N_sample, 
                        padding = padding,
                        stride =stride,
                        kernel_sz = kernel
                       
                        )
        block['act'] =  nn.ReLU()

        block['pool'] = nn.MaxPool1d( kernel_size = kernel) if max_pool else lambda x:x
        block['drop'] = nn.Dropout(0.1)
        block['Lout'] = get_out_size(   Lin = Lout, 
                            padding = 0,
                            stride = kernel,
                            kernel_sz =  kernel
                            ) if max_pool else Lout

        #print(Lout, block['Lout'])
        block['output_chan'] = N_output_chan
        block['batch_norm'] =  nn.BatchNorm1d(num_features=N_output_chan)
        return block
         
            

    def forward(self, x):
        # ( batch_sz, n_chan, n_sample)  = ( batch_sz,  7 , 500 )
        for  b in self.conv_blocks:
            x = b['drop'](b['act'](b['batch_norm'](b['pool'](b['conv'](x)))))
            
        x =  torch.cat(torch.split(x,1,dim=1),dim=2).squeeze(1) # stack & squeeze convs outputs
       
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return  x


class DreemModelMultihead(BaseModel):
    def __init__(self, conf, num_classes=2):
        super().__init__()
        self.N_samples = 500
        self.N_chan = 7
        
        self.conv_blocks = {}
        
   
        #build first layer
        self.conv_blocks["layer_1"]={"conv":nn.ModuleList([]),
                                    "act":nn.ModuleList([]),
                                    "pool":nn.ModuleList([]),
                                    "drop":nn.ModuleList([]),
                                    "batch_norm":nn.ModuleList([]),
                                    "n_filters":[],
                                    "Lout":[]
                                    }
        for idx, b_conf in enumerate(conf):
            self.build_block(b_conf,layer=1,max_pool=True)
            for module in ('conv','act','pool','drop','batch_norm'):
                  setattr(self,f"L1_{module}_{idx}",self.conv_blocks["layer_1"][module][idx])
      
        
        
        #build a second layer for each head-channel
        self.conv_blocks["layer_2"]={"conv":nn.ModuleList([]),
                                    "act":nn.ModuleList([]),
                                    "pool":nn.ModuleList([]),
                                    "drop":nn.ModuleList([]),
                                    "batch_norm":nn.ModuleList([]),
                                    "n_filters":[],
                                    "Lout":[]
                                    }
        for idx, b_conf in enumerate(conf):
                self.build_block(b_conf,
                                in_channels = self.conv_blocks["layer_1"]['n_filters'][idx],
                                Lin=self.conv_blocks["layer_1"]['Lout'][idx],
                                layer=2,
                                max_pool=True)
                for module in ('conv','act','pool','drop','batch_norm'):
                  setattr(self,f"L2_{module}_{idx}",self.conv_blocks["layer_2"][module][idx])


     

        in_fc1 = sum(np.array(self.conv_blocks["layer_2"]["n_filters"]) * np.array(self.conv_blocks["layer_2"]['Lout']) )
      
        self.fc1 = nn.Linear( in_fc1, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
    def build_block(self,conf,in_channels=None,Lin=None,layer=1,max_pool=True):
        block={}

        if not in_channels:
            in_channels = self.N_chan
            Lin = self.N_samples

        block['conv'] = nn.Conv1d(in_channels = in_channels, 
                        out_channels = conf['n_filters'],
                        kernel_size = conf['kernel'],
                        stride =  conf.get('stride',conf['kernel']), 
                        padding = conf.get('padding',0)
                        )
        Lout = get_out_size(   Lin = Lin, 
                        padding = conf.get('padding',0),
                        dilation = conf.get('dilation',1), 
                        stride = conf.get('stride',conf['kernel']), 
                        kernel_sz = conf['kernel']
                        )
        block['act'] =  nn.ReLU()
        block['drop'] = nn.Dropout(conf.get('dropout',0.1))
        block['pool'] = nn.MaxPool1d( kernel_size = conf.get('pool_kernel',2),stride= conf.get('pool_stride',2)) if max_pool else lambda x:x
        block['n_filters'] = conf['n_filters']
        block['Lout'] = get_out_size(   Lin = Lout, 
                                    padding = 0,
                                    dilation = 1, 
                                    stride = conf.get('pool_kernel',2), 
                                    kernel_sz = conf.get('pool_stride',2)
                                    ) if max_pool else Lout
        block['batch_norm'] =  nn.BatchNorm1d(num_features=block['n_filters'])
        for k,v in block.items():
          self.conv_blocks.get(f"layer_{layer}").get(k).append(v)


       

    def forward(self, x):
        X1 = [self.conv_blocks["layer_1"]['drop'][idx](
                  self.conv_blocks["layer_1"]['act'][idx](
                    self.conv_blocks["layer_1"]['batch_norm'][idx](
                      self.conv_blocks["layer_1"]['pool'][idx](
                        self.conv_blocks["layer_1"]['conv'][idx](x)
                        )
                      )
                    )
                  ) for idx in range(3)]
        #build a second layer for each channel
        X2 = [self.conv_blocks["layer_2"]['drop'][idx](
                  self.conv_blocks["layer_2"]['act'][idx](
                    self.conv_blocks["layer_2"]['batch_norm'][idx](
                      self.conv_blocks["layer_2"]['pool'][idx](
                        self.conv_blocks["layer_2"]['conv'][idx](X1[idx])
                        )
                      )
                    )
                  ) for idx in range(3)]
     
        X = [torch.cat(torch.split(x,1,dim=1),dim=2).squeeze(1) for x in X2]
        # stack & squeeze convs outputs
        x = torch.cat(X,dim=1)
    

        x = F.dropout(x, p=0.5, training=self.training)
  
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1) 
        return  x
