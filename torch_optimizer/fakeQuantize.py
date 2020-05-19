import torch
from torch.nn import Module
import torch.nn as nn

import math
from functools import partial
import numpy as np


class FakeQuantizer(nn.Module):
    r'''
    This FakeQuantizer is for all torch tensor
    * :attr:`quant_min` specifies the minimum allowable quantized value.
    * :attr:`quant_max` specifies the maximum allowable quantized value.
    '''
    def __init__(self, quant_min, quant_max):
        # if qint should be -64, 63
        # if quint should be 0, 127
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.min_val = None
        self.max_val = None
        self.averaging_constant = 0.01
        self.eps = torch.finfo(torch.float32).eps

        
    def update_min_max(self, x_orig):
        '''
        this class use global min max
        TODO: write perchaneel min max, if so we need to update the forward as well
        '''
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        if self.min_val is None or self.max_val is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = self.min_val
            max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = min_val + self.averaging_constant * (torch.min(x) - min_val)
            max_val = max_val + self.averaging_constant * (torch.max(x) - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig
    
    def calculate_scale_zero_point(self, min_val, max_val):
        """
        TODO: check the range
        shall we do symmentric?
        """
        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.zeros_like(max_val))
        scale = torch.ones(min_val.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int64)
        device = 'cuda' if min_val.is_cuda else 'cpu'
        
        qmin = self.quant_min
        qmax = self.quant_max

        scale = (max_val - min_val) / float(qmax - qmin)
        scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.max(zero_point, torch.tensor(qmin, device=device, dtype=zero_point.dtype))
        zero_point = torch.min(zero_point, torch.tensor(qmax, device=device, dtype=zero_point.dtype))
        return scale, zero_point
    
    def forward(self, x):
        pass
        self.update_min_max(x)
        _scale, _zero_point = self.calculate_scale_zero_point()
        # TODO: check device
        self.zero_point = _zero_point
        self.scale = _scale
        X = torch.fake_quantize_per_tensor_affine(X, float(self.scale), int(self.zero_point), self.quant_min, self.quant_max)
        return X
    

# quantize_per_tensor

# np version in case    
def fake_quantize_affine(x, scale, zero_point, q_max, q_min, nbits = 8):
    # function for general cases
#     scale = (q_max - q_min) / 2**nbits
    x = np.clip(x, q_max, q_min)
    x_I = int((x - zero_point)/scale)
    return x_I * scale + zero_point