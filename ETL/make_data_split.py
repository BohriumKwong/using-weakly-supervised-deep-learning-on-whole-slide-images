# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:51:09 2020

@author: Bohrium.Kwong
"""
import os
import numpy as np
import glob
import gc
import pickle
from sklearn.model_selection import KFold

def normal_data_spilt(db_lib_path,save_new_db_lib_path):
    #简易划分数据集的方法,现已基本不用
    with open(db_lib_path, 'rb') as fp:
        train_data_lib = pickle.load(fp)
        train_slides_list = train_data_lib['slides']
#        train_flag = len(train_slides_list)
#        train_grids_list = train_data_lib['grid']
    #    train_sum = np.sum([len(x) for x in train_grids_list])
        train_targets_list = train_data_lib['targets']
#        train_patch_size_list = train_data_lib['patch_size']
        train_batch_info_list = train_data_lib['batch']
    
    target_train_slide = []
    target_val_slide = []
    target_train = []
    target_val = []
    for i in range(len(train_batch_info_list)):
        if train_batch_info_list[i] in ['batch_1_SYSUCC','batch_2_SYSUCC','batch_3_SYSUCC']:
            if np.random.binomial(1,0.275):
                #使用random进行具体划分
                target_val.append(train_targets_list[i])
                target_val_slide.append(train_slides_list[i])
            else:
                target_train.append(train_targets_list[i])
                target_train_slide.append(train_slides_list[i])

    print(np.sum(target_train),np.sum(target_val))
    print(len(target_train),len(target_val))
    #fold_1
    #74 34
    #365 155
    #fold_2
    #81 27
    #385 135
    #fold_3
    #79 29
    #392 128
    with open(save_new_db_lib_path, 'wb') as fp:
        pickle.dump((target_train_slide,target_val_slide), fp)
    #normal_data_spilt('./output/lib/512/cancer_mask_sample/PAIP_train_data_lib.db','./output/lib/tum_region_mid/SYSUCC_sample_train_val.db')
##上述是比较简单直接的针对SYSUCC划分数据集的方法，最终保存的结果以划分的数据集中slide的MSI和MSS比例和整体数据集差不多一致就行
# 需要注意的是，这种划分数据集的方法并非是K-fold的方法

    

#mss_count = 0
#msi_count = 0
#for i in range(len(train_slides_list)):
#    if train_slides_list[i] in val_dset.slidenames:
#        if train_targets_list[i]==0:
#            mss_count += len(train_grids_list[i])
#        else:
#            msi_count += len(train_grids_list[i])
#print(mss_count,msi_count)


def make_tcga_data_lib(train_dir,origin_db_path,save_new_db_lib_dir):
    #针对TCGA结构的数据集生成对应的db文件,即目录结构是root_dir/slide_name/slide_name_(grid_x,grid_y).jpg的形式
    # 该方法是根据原有生成的TCGA的db文件(自己截图)对新数据(官方提供)生成新的db文件的方法
    file_list = os.listdir(train_dir)
    #'TCGA-CRC-DX-Slide'
    batch_select = ['batch_4_TCGA_analysis']
    slides_list = []
    grid_list = []
    targets_list = []
    batch_list = []
    with open(origin_db_path, 'rb') as fp:
        lib = pickle.load(fp)

    for i,slide in enumerate(lib['slides']):
        if lib['batch'][i] in batch_select:
            slides_list.append(slide)
            grid_list.append(lib['grid'][i])
            targets_list.append(lib['targets'][i])
            batch_list.append(lib['batch'][i])
    keep_list = []
    #keep_origin_list = []
    for file_name in file_list:
        if file_name.split('.')[0] in slides_list:
            keep_list.append(file_name.split('.')[0])
    #原来的TCGA中slide的名字太长(.后面其实可以不要),现重新提取slide名字进行保存

    with open('', 'rb') as fp:
        lib = pickle.load(fp)
        # 'output/lib/tum_region_mid/cnn_train_data_lib.db'

    new_slides_list = []
    new_grid_list = []
    new_targets_list = []
    new_batch_list = []
    grid_len_lst = []
    for i,slide in enumerate(lib['slides']):
        if slide in keep_list:
    #        new_grid_list.append(lib['grid'][i])
            tmp_grid_list=glob.glob(os.path.join(train_dir,slide+'*',slide+'*jpg'))
            grid_len_lst.append(len(tmp_grid_list))
            if len(tmp_grid_list) > 0 :
                tmp_grid = [(int(os.path.basename(x).split('(')[1].split(')')[0].split(',')[0]), int(os.path.basename(x).split('(')[1].split(')')[0].split(',')[1])) 
                for x in tmp_grid_list]
                new_slides_list.append(os.path.basename(tmp_grid_list[0]).split('_(')[0])
                #因为数据集文件名是slide_name_(grid_x,grid_y).jpg的形式,所以要根据"("和")"来提取具体的坐标值
                new_grid_list.append(tmp_grid)
                del tmp_grid_list,tmp_grid
                gc.collect()
                new_targets_list.append(lib['targets'][i])
                new_batch_list.append(lib['batch'][i])


#    kf = KFold(n_splits=3)
#    for train,val in  kf.split(new_slides_list):
#        target_train,target_val = [],[]
#        for i in range(len(new_slides_list)):
#            if i in train:
#                target_train.append(new_targets_list[i])
#            elif i in val:
#                target_val.append(new_targets_list[i])
#        print(np.sum(target_train),np.sum(target_val))
#        print(len(target_train),len(target_val))
    #FOLD 1
    #51 26
    #294 147
    #FOLD 2
    #49 28
    #294 147
    #FOLD 3
    #54 23
    #294 147
    # 上述这种是简单的使用kFold进行划分的方法，但在sklearn源码中,kFold只是根据样本顺序来划分,\
    #   对于label分布不均匀的数据集来说不建议直接采样上述的方法
    
    
#    def k_fold_spilt(k,index_list_len):
#        train,val = [],[]
#        if index_list_len > k**2:
#            for i in range(index_list_len):
#                if i % k ==0:
#                    val.append(i)
#                else:
#                    train.append(i)
#        return train,val

    kf = KFold(n_splits=3)
    shuffle_slide_list = np.random.choice(new_slides_list,len(new_slides_list),replace=False)
    # 打乱之后再调用kFold方法进行划分数据集,有可能会得到更好的结果
    
    for k,(train,val) in enumerate(kf.split(shuffle_slide_list)):
        target_train,target_val = [],[]
        target_train_slide,target_val_slide = [],[]
        train_list = [shuffle_slide_list[x] for x in train]
        val_list = [shuffle_slide_list[x] for x in val]
        train_sum_msi,train_sum_mss,val_sum_msi,val_sum_mss  = 0,0,0,0
        for i,slide in enumerate(new_slides_list):
            if slide in train_list:
                target_train.append(new_targets_list[i])
                target_train_slide.append(new_slides_list[i])
                if new_targets_list[i] == 1:
                    train_sum_msi += len(new_grid_list[i])
                else:
                    train_sum_mss += len(new_grid_list[i])
            else:
                target_val.append(new_targets_list[i])
                target_val_slide.append(new_slides_list[i])
                if new_targets_list[i] == 1:
                    val_sum_msi += len(new_grid_list[i])
                else:
                    val_sum_mss += len(new_grid_list[i])
        print("FOLD"+str(k+1)+":")
        print(np.sum(target_train),np.sum(target_val))
        print(train_sum_msi,val_sum_msi)
        print(len(target_train)-np.sum(target_train),len(target_val)-np.sum(target_val))
        print(train_sum_mss,val_sum_mss)
        with open(save_new_db_lib_dir,str(k+1) + '_3.db', 'wb') as fp:
            pickle.dump((target_train_slide,target_val_slide), fp)