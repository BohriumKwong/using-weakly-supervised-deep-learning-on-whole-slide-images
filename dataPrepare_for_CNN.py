import torch
import openslide
#from openslide.deepzoom import DeepZoomGenerator
import os
import pandas as pd
#import random
import numpy as np
import cv2
import time 
import glob
import copy

# ---------------------- 相关变量的格式定义，参考 README.md ---------------------- #
# 最终保存全部数据的字典
train_data_lib = {}
train_slides_list = []   # 存储文件路径
train_targets_list = []  # 存储目标信息
train_grids_list = []    # 存储格点信息
train_patch_size_list = [] #存储WSI文件对应切片尺寸

val_data_lib = {}
val_slides_list = []   # 存储文件路径
val_targets_list = []  # 存储目标信息
val_grids_list = []    # 存储格点信息
val_patch_size_list = [] #存储WSI文件对应切片尺寸

level = 0          # 使用 openslide 读取时的层级，默认表示以最高分辨率
patch_size = 512   # 默认的切片的尺寸

train_flag = 0
val_flag = 0
train_sum = 0
val_sum = 0

target_file = '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/MSI_MUT/output_process/MIL_ETL/target.xlsx'


batch_list = ['batch_1','batch_2','batch_3','batch_4']
data_path_list = ['/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_1_SYSUCC',
                  '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_2_SYSUCC',
                  '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_3_SYSUCC',
                  '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_4_TCGA_analysis']
#npy_list = ['/cptjack/totem_data_backup/totem/COLORECTAL_DATA/1_region_prediction_npy_nonorm',
#            '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/TSP-CNN-colorectal/output/region_prediction_npy_nonorm/batch_2_SYSUCC',
#            '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/TSP-CNN-colorectal/output/region_prediction_npy_nonorm/batch_3_SYSUCC',
#            '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/TSP-CNN-colorectal/output/region_prediction_npy_nonorm/batch_4_TCGA']

for i, batch in enumerate(batch_list):
    data_df = pd.read_excel(target_file,sheet_name = batch,sep='')
    data_df['dataset'] = batch
    data_df['wispath'] = ''
    data_df['delete'] = ''
    data_df['sample_count'] = ''
    for index,_ in data_df.iterrows():
        
        basename = os.path.basename(str(data_df.iloc[index,0])).split('.')[0]
        wispath = glob.glob(os.path.join(data_path_list[i], "*" + basename + "*svs"))
        if len(wispath)==1 :
            data_df.iloc[index,3] = wispath[0]
# ---------------------- 开始处理数据，获取 lib ---------------------- #
            start_time = time.time()
            slide = openslide.open_slide(wispath[0])

            try:
                properties = slide.properties
                mpp = np.float(properties['openslide.mpp-x'])
                if round(mpp/0.25) == 1:
                    patch_size = 1024
                elif round(mpp/0.25) == 2:
                    patch_size = 512
                else:
                    patch_size = 256
            except Exception:
                    patch_size = 512
            
            try:
                svs_level = 2 if len(slide.level_dimensions) >2 else len(slide.level_downsamples) -1
                thumb = slide.get_thumbnail(slide.level_dimensions[svs_level])
            except Exception:
                print(basename + ' TIFFRGBAImageGet failed!' )
                data_df.iloc[index,4] = 'y'
                #get_thumbnail执行失败的话将删除标记置为"Y"并跳出当前循环
                continue
            #先尝试以level=2获取全片截图,如果有异常就将level适配文件的设置再提取截图
            thumb = np.array(thumb)[:,:,:3]
            thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
            thumb = cv2.GaussianBlur(thumb, (5,5), 0)   
            thumb = cv2.threshold(thumb,240, 255, cv2.THRESH_BINARY_INV)[1]
            #thumb是过滤背景区域的二值图，>0的地方表示组织区域
            level_downsamples = round(slide.level_downsamples[svs_level])
#            sample_pro = min(1 - np.sum(thumb>0)/thumb.shape[0]/thumb.shape[1],0.99)
            sample_pro = min(7000 *(patch_size/level_downsamples)**2/np.sum(thumb>0),1)
            #sample_pro是基于组织区域占比计算出来的采样率,为了控制采样数量,需要在采样前进行概率判断,\
            # 注释的方法不是适用于尺寸差异太大的图片(只能根据个体比例调整,但如果出现尺寸超大的图,依然会采出很多图片,不利于训练)。\
            # 目前的方法是设定一个具体的上限数字(如7000),如果silde有效区域/patch数量>7000就100%采样,\
            #  否则就以7000/(silde有效区域/patch数量)来作为采样概率。
            # 原则上组织区域越大的图片每一块截图采样的概率就会调低,控制整体样本数量的同时又能做到均衡采样。
            w, h = slide.dimensions            
            
        #        if random.randint(0, 21) < 14:
            if batch != 'batch_4':
                train_flag = train_flag + 1
               
                cur_patch_cords = []
        
                for j in range(0, h, patch_size):
                    for k in range(0, w, patch_size):
                        bottom = int(j / level_downsamples)
                        top = bottom + int(patch_size / level_downsamples) -1
                        left = int(k / level_downsamples)
                        right = left + int(patch_size / level_downsamples) -1
                        #前期先以背景/组织区域作为过滤条件，后期可以用区域分类结果作为过滤条件
                        if np.sum(thumb[bottom:top,left:right] > 0) > 0.60 * (patch_size / level_downsamples)**2 \
                                  and np.random.binomial(1,sample_pro):
                            cur_patch_cords.append((k,j))
                data_df.iloc[index,5] = len(cur_patch_cords)
                if len(cur_patch_cords) > 0:
                    train_slides_list.append(wispath[0])
                    train_patch_size_list.append(patch_size)
                    train_targets_list.append(data_df.iloc[index,1])
                    train_grids_list.append(cur_patch_cords)
                    train_sum = train_sum + len(cur_patch_cords)
            else:
                val_flag = val_flag + 1
        
                cur_patch_cords = []
        
                for j in range(0, h, patch_size):
                    for k in range(0, w, patch_size):
                        bottom = int(j / level_downsamples)
                        top = bottom + int(patch_size / level_downsamples) -1
                        left = int(k / level_downsamples)
                        right = left + int(patch_size / level_downsamples) -1
                        # 根据当前循环移动到的patch所在的区域，在上述提取的背景/组织区域二值图中映射同样的位置,当覆盖的像素点的占比超过64%时,、
                        #   说明该patch所在是我们感兴趣的区域,可以进行patch坐标记录
                        if np.sum(thumb[bottom : top,left : right ] > 0) > 0.64 * (patch_size / level_downsamples)**2 \
                                  and np.random.binomial(1,sample_pro):
                            cur_patch_cords.append((k,j))
                data_df.iloc[index,5] = len(cur_patch_cords)
                if len(cur_patch_cords) > 0:
                    val_slides_list.append(wispath[0])
                    val_patch_size_list.append(patch_size)
                    val_targets_list.append(data_df.iloc[index,1])            
                    val_grids_list.append(cur_patch_cords)
                    val_sum = val_sum + len(cur_patch_cords)
                    
            slide.close()    
            print(basename + ' get %d cords, needed %.2f sec.' % (len(cur_patch_cords),time.time() - start_time))

        else:
            data_df.iloc[index,4] = 'y'
    if i == 0:
        whole_data = copy.deepcopy(data_df)
    else:
        whole_data = pd.concat([whole_data,data_df],axis=0)            
            
print('%d WSI files ,total %d samples in train data set.' %(train_flag,train_sum))
print('%d WSI files ,total %d samples in val data set.' %(val_flag,val_sum))


train_data_lib['slides'] = train_slides_list
train_data_lib['grid'] = train_grids_list
train_data_lib['targets'] = train_targets_list
train_data_lib['level'] = level
train_data_lib['patch_size'] = train_patch_size_list
if not os.path.isdir('./output/lib/512'): os.makedirs('./output/lib/512')
torch.save(train_data_lib, './output/lib/512/all_region/cnn_train_data_lib.db')

val_data_lib['slides'] = val_slides_list
val_data_lib['grid'] = val_grids_list
val_data_lib['targets'] = val_targets_list
val_data_lib['level'] = level
val_data_lib['patch_size'] = val_patch_size_list
torch.save(val_data_lib, 'output/lib/512/all_region/cnn_val_data_lib.db')

whole_data = whole_data.drop(['file_name'],axis =1) 
whole_data = whole_data[whole_data['delete']!='y']
whole_data = whole_data.drop(['delete'],axis =1)
whole_data.to_excel('all_region_512.xlsx',index = False)