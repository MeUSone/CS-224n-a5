#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self,input_size):
        super(Highway,self).__init__()
        self.proj=nn.Linear(in_features=input_size,out_features=input_size)
        self.gate=nn.Linear(in_features=input_size,out_features=input_size)

    def forward(self,input):
        proj=F.relu(self.proj(input))
        gate=F.sigmoid(self.gate(input))
        highway=gate*proj+(1-gate)*input
        return highway

### END YOUR CODE 

