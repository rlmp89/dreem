

from torchaudio.transforms import Spectrogram
from utils.util import band_filter
import json

class spectrogram(object):
    from torchaudio.transforms import Spectrogram
    """Apply spectrogram
    Args: 
        nfft: n sample
    """
    def __init__(self, nfft):
        self.spectro = Spectrogram(nfft,normalized=True,power=2)
    def __call__(self, sample):
        return  self.spectro(sample)
    def __name__(self):
        return "spectrogram"


class minmaxscaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    eps = 1.e-6

    def __call__(self, tensor):
        _max = tensor.max(dim=1, keepdim=True)[0]
        _min = tensor.min(dim=1, keepdim=True)[0]
        scale = 1.0 / (_max - _min  + eps ) 
        tensor.mul_(scale).sub_(_min)
        return tensor
    def __name__(self):
        return "min-max scaler"


class outliers(object):
    """remove outliers (bad trials)
    """
    def __init__(self, path=''):
        if path:
            self.out_list = json.load(open(path,'r'))['out']
        self.outliers=[]
    def __call__(self, n_idv,training):
        if training:
            for idx, o in enumerate(self.out_list):
                for bad_trial in o:     
                    self.outliers.append(idx + n_idv*bad_trial)
            self.outliers = sorted(self.outliers) 
        return  self.outliers
    def __name__(self):
        return "outliers removal"