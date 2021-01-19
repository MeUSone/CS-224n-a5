#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size=5):
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
    def forward(self,input):
        before_pool=F.relu(self.conv(input))
        conv_out=self.maxpool(before_pool).squeeze(dim=2)
        return conv_out
### END YOUR CODE

