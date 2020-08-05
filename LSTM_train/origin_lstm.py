# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:57:50 2020

@author: Bohrium Kwong
"""

#import torch
import torch.nn as nn
import torch.nn.functional as F



class ori_lstm(nn.Module): 
    def __init__(self,input_size,hidden_size,num_layer,batch_first,output_size): 
        super(ori_lstm,self).__init__() 
        
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size, 
                           num_layers=num_layer, 
                           batch_first=True
                           )
        self.out = nn.Sequential(nn.Linear(hidden_size,output_size),nn.Softmax())
        
    def forward(self, x): 
        r_out,(h_n,h_c) = self.rnn(x,None) # x (batch,time_step,input_size) 
        out = self.out(h_n[-1,:,:]) #(batch,time_step,input)
        return out

