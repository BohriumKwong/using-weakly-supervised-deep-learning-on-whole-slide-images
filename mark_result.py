#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:53:14 2020

@author: root
"""
import pandas as pd
import os
import numpy as np


def result_excel(dset,pred,save_name):
    file_list = []
    for i in dset.slidenames:
       file_list.append(os.path.basename(i)) 
    
    c = {'file_name':file_list,'predict':list(np.argmax(pred,axis=1)),'true_label':dset.targets}
    test = pd.DataFrame(c)
    save_dir = os.path.join('output','result')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    test.to_excel(os.path.join(save_dir,save_name +'.xlsx'),index = False)
    
def result_excel_origin(dset,pred,save_name):
    file_list = []
    for i in dset.slidenames:
       file_list.append(os.path.basename(i)) 
    
    c = {'file_name':file_list,'predict':list(pred),'true_label':dset.targets}
    test = pd.DataFrame(c)
    save_dir = os.path.join('output','result')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    test.to_excel(os.path.join(save_dir,save_name +'.xlsx'),index = False)