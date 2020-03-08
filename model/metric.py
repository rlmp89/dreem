import torch
import numpy as np
import pandas as pd
from functools import partial
import sys
this = sys.modules[__name__]
#################################
# decorators
#################################
def globalMetric(metric):
    '''
    tags metric that cannot be averaged across batches
    '''
    metric.global_metric = True
    return metric

def trainMetric(metric):
    '''
    tags metric that can be used during training
    '''
    metric.train_metric=True
    return metric

def validMetric(metric):
    '''
    tags metric that can be used during validation
    '''
    metric.valid_metric=True
    return metric
#################################

#set metric wrapper for metrics with args
def wrappedMetric(metric_name, args={}):
    target_func = getattr(this,metric_name)
    metric = partial(target_func,**args)
    # set attributes inherited from target function
    metric.__name__= target_func.__name__
    metric.__name__ += "_"+"_".join(["_".join([str(k),str(v)]) for k,v in args.items()])
  
    for k,v in target_func.__dict__.items():
      setattr(metric,k,v)
    return metric

class WrappedMetric(object):
    def __init__(self,met):
        if type(met)==str:
            metric_name = met
            self.kwargs={}
        else:
            metric_name = met['type']
            self.kwargs = met['args']

        self.target_func = getattr(this,metric_name)        
        self.metric = partial(self.target_func,**self.kwargs)
        # set attributes inherited from target function
        for k,v in self.target_func.__dict__.items():
            setattr(self.metric,k,v)
    def __name__(self):
        n = self.target_func.__name__
        n += "_"+"_".join(["_".join([str(k),str(v)]) for k,v in self.kwargs.items()])
        return n

    def __call__(self,output, target):
        return self.metric(output,target)


@trainMetric
@validMetric
def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)




@validMetric
@globalMetric
def roc_auc(output, target):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This metric requires sklearn to be installed.")
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_pred = torch.argmax(output, dim=1).cpu().numpy()
    return roc_auc_score(y_true, y_pred)

@globalMetric
def dreem_merged_accuracy(output, target):
    with torch.no_grad():
        # dim= (n_individuals, 40, n_classes)  
        #calculate prediction mean among 40 trials for each individual
        y_pred_merged = torch.stack(output.chunk(40),axis=1).mean(axis=1)  
        # dim = (n_individuals, n_classes)
        y_true_merged = torch.stack(target.chunk(40),axis=1)[:,0]
        return accuracy(y_pred_merged, y_true_merged)  


@trainMetric
@validMetric
def f1(output, target):
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise RuntimeError("This metric requires sklearn to be installed.")
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_pred = torch.argmax(output, dim=1).cpu().numpy()
    return f1_score(y_true, y_pred,average="weighted")

@trainMetric
@validMetric
def cohen_kappa(output, target):
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        raise RuntimeError("This metric requires sklearn to be installed.")
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_pred = torch.argmax(output, dim=1).cpu().numpy()
    return cohen_kappa_score(y_true, y_pred)



@validMetric
@globalMetric
def accuracy_bootstrap_lowerbound(output, target,K,sigma=2):
    assert sigma in (1,2,3), "Sigma Confidence interval  must be in (1,2,3)"
    sz = output.size()[0]
    all_acc= []
    from random import choices
    from math import sqrt
    with torch.no_grad():
        for _ in range(K):
            bootstrap_idx = choices(list(range(sz)),k = sz)
            o = output[bootstrap_idx,:]
            t = target[bootstrap_idx]
            all_acc.append(accuracy(o, t) )
        all_acc = np.array(all_acc)
        mu= all_acc.mean()
        std = all_acc.std()
        return mu - sigma*std/sqrt(sz)