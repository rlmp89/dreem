import torch.nn.functional as F
import torch.tensor as T

def nll_loss(output, target, weight=None):
    w = T(weight).float().to('cuda') if weight else None
    return F.nll_loss(output, target,weight=w)

