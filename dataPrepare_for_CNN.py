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

# svs 文件所在路径
#data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."), 'dataset')
data_dir = '/your path/svs'

# target 列表
target_df = pd.read_csv('/yourpath/target.csv')
#target_df是一个含有两列的表,格式如下:
#| file_name        | class |
#| :--------------: | :---: |
#| ---------*.svs   |   0   |
#| --------/*.svs   |   1   |
#| ..............   |  ...  |


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
patch_size = 224   # 默认的切片的尺寸

train_flag = 0
val_flag = 0
train_sum = 0
val_sum = 0

files = glob.glob(os.path.join(data_dir, "*.svs"))
files = sorted(files)

# ---------------------- 开始处理数据，获取 lib ---------------------- #
for filename in files:
#    if not filename.endswith('svs'):
#        continue
    start_time = time.time()
    slide = openslide.open_slide(filename)
    basename = os.path.basename(filename)
    try:
        properties = slide.properties
        mpp = np.float(properties['openslide.mpp-x'])
        if round(mpp/0.25) == 1:
            patch_size = 448
        elif round(mpp/0.25) == 2:
            patch_size = 224
        else:
            patch_size = 112
    except Exception:
            patch_size = 224
    
    try:
        svs_level = 2
        thumb = slide.get_thumbnail(slide.level_dimensions[svs_level])
    except Exception:
        svs_level = len(slide.level_downsamples) -1
        thumb = slide.get_thumbnail(slide.level_dimensions[svs_level])
    #先尝试以level=2获取全片截图,如果有异常就将level适配文件的设置再提取截图
    thumb = np.array(thumb)[:,:,:3]
    thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    thumb = cv2.GaussianBlur(thumb, (5,5), 0)   
    thumb = cv2.threshold(thumb,240, 255, cv2.THRESH_BINARY_INV)[1]
    #thumb是过滤背景区域的二值图，>0的地方表示组织区域
    sample_pro = min(1 - np.sum(thumb>0)/thumb.shape[0]/thumb.shape[1],0.99)
    # sample_pro是基于组织区域占比计算出来的采样率,为了控制采样数量,需要在采样前进行概率判断,、
    # 原则上组织区域越大的图片每一块截图采样的概率就会调低,控制整体样本数量的同时又能做到均衡采样。
    w, h = slide.dimensions
    level_downsamples = round(slide.level_downsamples[svs_level])
    
#        if random.randint(0, 21) < 14:
    if np.random.binomial(1,0.72):
        train_flag = train_flag + 1
       
        cur_patch_cords = []

        for j in range(0, h, patch_size):
            for i in range(0, w, patch_size):
                bottom = int(j / level_downsamples)
                top = bottom + int(patch_size / level_downsamples) -1
                left = int(i / level_downsamples)
                right = left + int(patch_size / level_downsamples) -1
                #前期先以背景/组织区域作为过滤条件，后期可以用区域分类结果作为过滤条件
                if np.sum(thumb[bottom:top,left:right] > 0) > 0.60 * (patch_size / level_downsamples)**2 \
                          and np.random.binomial(1,sample_pro):
                    cur_patch_cords.append((i,j))
        if len(cur_patch_cords) > 5:
            train_slides_list.append(filename)
            train_patch_size_list(patch_size)
            train_targets_list.append(target_df[target_df['slide'] == basename]['target'].values[0])
            train_grids_list.append(cur_patch_cords)
            train_sum = train_sum + len(cur_patch_cords)
    else:
        val_flag = val_flag + 1

        cur_patch_cords = []

        for j in range(0, h, patch_size):
            for i in range(0, w, patch_size):
                bottom = int(j / level_downsamples)
                top = bottom + int(patch_size / level_downsamples) -1
                left = int(i / level_downsamples)
                right = left + int(patch_size / level_downsamples) -1
                # 根据当前循环移动到的patch所在的区域，在上述提取的背景/组织区域二值图中映射同样的位置,当覆盖的像素点的占比超过64%时,、
                #   说明该patch所在是我们感兴趣的区域,可以进行patch坐标记录
                if np.sum(thumb[bottom : top,left : right ] > 0) > 0.64 * (patch_size / level_downsamples)**2 \
                          and np.random.binomial(1,sample_pro):
                    cur_patch_cords.append((i,j))
        if len(cur_patch_cords) > 5:
            val_slides_list.append(filename)
            val_patch_size_list(patch_size)
            val_targets_list.append(target_df[target_df['slide'] == basename]['target'].values[0])            
            val_grids_list.append(cur_patch_cords)
            val_sum = val_sum + len(cur_patch_cords)
        
    print(filename + ' get %d cords, needed %.2f sec.' % (len(cur_patch_cords),time.time() - start_time))
        
print('%d WSI files ,total %d samples in train data set.' %(train_flag,train_sum))
print('%d WSI files ,total %d samples in val data set.' %(val_flag,val_sum))


train_data_lib['slides'] = train_slides_list
train_data_lib['grid'] = train_grids_list
train_data_lib['targets'] = train_targets_list
train_data_lib['level'] = level
train_data_lib['patch_size'] = train_patch_size_list
if not os.path.isdir('./output/lib'): os.makedirs('./output/lib')
torch.save(train_data_lib, './output/lib/cnn_train_data_lib.db')

val_data_lib['slides'] = val_slides_list
val_data_lib['grid'] = val_grids_list
val_data_lib['targets'] = val_targets_list
val_data_lib['level'] = level
val_data_lib['patch_size'] = val_patch_size_list
torch.save(val_data_lib, 'output/lib/cnn_val_data_lib.db')