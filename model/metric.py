import torch
import numpy as np
import pandas as pd
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
