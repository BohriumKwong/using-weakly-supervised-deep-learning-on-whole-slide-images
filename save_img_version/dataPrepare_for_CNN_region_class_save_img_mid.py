import os
import pandas as pd
#import random
import numpy as np
import time 
import glob
import copy
import sys
sys.path.append('../')
import PIL.Image as Image
from output_process.process_script import matrix_resize
from utils.openslide_utils import Slide
import pickle


def etl_process(target_file,batch_list,data_path_list,npy_list,image_save_dir):
    """
    Parameters
    ----------
    target_file:  str
        the path of all WSIs's label information file(excel format)
    batch_list:  list
        the list of batch name of WSIs
    data_path_list:  list
        the list of WSIs file path
    npy_list(TCGA_USELESS):  list
        the list of WSIs's region classification result file path(npy format)
    image_save_dir: str
        the path that save images cut from WSIs 
    """
    level = 0          # 使用 openslide 读取时的层级，默认表示以最高分辨率
    patch_size = 512  # 默认的切片的尺寸
    
    
    if not os.path.exists('./output/lib/512/tum_region_mid/cnn_train_data_lib.db'):
        # 最终保存全部数据的字典
        train_data_lib = {}
        train_slides_list = []   # 存储文件路径
        train_targets_list = []  # 存储目标信息
        train_grids_list = []    # 存储格点信息
        train_patch_size_list = [] # 存储WSI文件对应切片尺寸
        train_batch_info_list = []
        train_flag = 0           # 记录已采集slide的数量
        train_sum = 0            # 记录已采集的总切图数

    else:
        with open('./output/lib/512/tum_region_mid/cnn_train_data_lib.db', 'rb') as fp:
           train_data_lib = pickle.load(fp)
           train_slides_list = train_data_lib['slides']
           train_flag = len(train_slides_list)
           train_grids_list = train_data_lib['grid']
           train_sum = np.sum([len(x) for x in train_grids_list])
           train_targets_list = train_data_lib['targets']
           train_patch_size_list = train_data_lib['patch_size']
           train_batch_info_list = train_data_lib['batch']
    # 如果已有有效的lib.db文件,则相关list直接读入文件而不是重零开始计算    

    def save_lib():            
        train_data_lib['slides'] = train_slides_list
        train_data_lib['grid'] = train_grids_list
        train_data_lib['targets'] = train_targets_list
        train_data_lib['level'] = level
        train_data_lib['patch_size'] = train_patch_size_list
        train_data_lib['batch'] = train_batch_info_list
        if not os.path.isdir('./output/lib/512/tum_region_mid'): os.makedirs('./output/lib/512/tum_region_mid')
#        torch.save(train_data_lib, './output/lib/512/all_region/sample_pro_' + str(patch_widen_value) + '_cnn_train_data_lib.db')
        with open('./output/lib/512/tum_region_mid/cnn_train_data_lib.db', 'wb') as fp:
            pickle.dump(train_data_lib, fp)        
        # 内部方法,用于每一次slide完成后及时存储当前信息，防止由于意外中断而导致之前运行的记录没有保存
        
    for i, batch in enumerate(batch_list):
        data_df = pd.read_excel(target_file,sheet_name = batch,sep='')
    #    if i < 3:
    #        dataset = 'train'
    #    else:
    #        dataset = 'val'
    
        data_df['dataset'] = batch
        data_df['wispath'] = ''
        data_df['npypath'] = ''
        data_df['delete'] = ''
        data_df['sample_count'] = ''
        for index,_ in data_df.iterrows():
            
            basename = os.path.basename(str(data_df.iloc[index,0])).split('.')[0]
            wispath = glob.glob(os.path.join(data_path_list[i], "*" + basename + "*svs"))
            npypath = glob.glob(os.path.join(npy_list[i], "*" + basename + "*output.npy"))
                
            if len(wispath)==1 and len(npypath) ==1 and basename not in train_slides_list:
                data_df.iloc[index,3] = wispath[0]
                data_df.iloc[index,4] = npypath[0]
    
            # ---------------------- 开始处理数据，获取 lib ---------------------- #
                start_time = time.time()
                slide = Slide(wispath[0])
                try:
                    if round(slide.get_mpp()/0.00025) == 1:
                        patch_size = 448
                    elif round(slide.get_mpp()/0.00025) == 2:
                        patch_size = 224
                    else:
                        patch_size = 112
                except Exception:
                    patch_size = 224
                    
                thumb = np.load(npypath[0])
                if len(thumb.shape)==3:
                    if thumb.shape[2]>1:
                        thumb = np.argmax(thumb,axis=2)
                        thumb = np.uint8(thumb)
                # 如果加载的分类结果矩阵是维度>2且通道数>1,说明这个矩阵是N分类概率矩阵,需要进行argmax处理还原为常用的分类结果矩阵
                try:              
                    cur_patch_cords = []                
                    for j in range(thumb.shape[0]):
                        for k in range(thumb.shape[1]):
                        #这里以区域分类结果作为过滤条件,8即TUM区域
                            if thumb[j,k] == 8:
                                k_cor = int(k*patch_size + patch_size/2)
                                j_cor = int(j*patch_size + patch_size/2)
                                cur_patch_cords.append((k_cor,j_cor))
                                if not os.path.exists(os.path.join(image_save_dir,str(224),data_path_list[i].split("/")[-1],str(data_df.iloc[index,1]),\
                                                        basename + '_' + str(k_cor) + '_' + str(j_cor)+'.jpg')):
                                    img = slide.read_region((int(k_cor-112),int(j_cor-112)),0,(224,224)).convert('RGB')                              
                                    img.save(os.path.join(image_save_dir,str(224),data_path_list[i].split("/")[-1],str(data_df.iloc[index,1]),\
                                                            basename + '_' + str(k_cor) + '_' + str(j_cor)+'.jpg'))
                                if not os.path.exists(os.path.join(image_save_dir,str(448),data_path_list[i].split("/")[-1],str(data_df.iloc[index,1]),\
                                                        basename + '_' + str(k_cor) + '_' + str(j_cor)+'.jpg')):
                                    img = slide.read_region((int(k_cor-224),int(j_cor-224)),0,(448,448)).convert('RGB') 
                                    img = img.resize((224,224),Image.BILINEAR)
                                    img.save(os.path.join(image_save_dir,str(448),data_path_list[i].split("/")[-1],str(data_df.iloc[index,1]),\
                                                            basename + '_' + str(k_cor) + '_' + str(j_cor)+'.jpg'))

                    data_df.iloc[index,6] = len(cur_patch_cords)
                    if len(cur_patch_cords) >0:
                        train_flag = train_flag + 1                                
                        train_slides_list.append(basename)
                        train_patch_size_list.append(patch_size)
                        train_batch_info_list.append(data_path_list[i].split("/")[-1])
                        train_targets_list.append(data_df.iloc[index,1])
                        train_grids_list.append(cur_patch_cords)
                        train_sum = train_sum + len(cur_patch_cords)
        
                        save_lib()
                    # 每处理一张图片就保存一次当前的进度
                    slide.close()
                    print(basename + ' get %d cords, needed %.2f sec.' % (len(cur_patch_cords),time.time() - start_time))
                
                except Exception:
                    data_df.iloc[index,5] = 'y' 
                    print(basename + ' failed.')
            else:
                data_df.iloc[index,5] = 'y'
        if i == 0:
            whole_data = copy.deepcopy(data_df)
        else:
            whole_data = pd.concat([whole_data,data_df],axis=0)
                    
    
        whole_data = whole_data.drop(['file_name'],axis =1) 
        whole_data = whole_data[whole_data['delete']!='y']
        whole_data = whole_data.drop(['delete'],axis =1)
        whole_data.to_excel('tum_mid_etl_tag_512.xlsx',index = False)
        
    print('%d WSI files ,total %d samples in train data set.' %(train_flag,train_sum))
        

if __name__ == '__main__':
    image_save_dir = '/cptjack/totem_disk/totem/colon_pathology_data/MIL_202005'
    if not os.path.isdir(image_save_dir): os.makedirs(image_save_dir)
    # target 列表
    target_file = '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/MSI_MUT/output_process/MIL_ETL/target.xlsx'
    
    npy_sub_dir = '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/TSP-CNN-colorectal/output/region_prediction_npy_nonorm/torch/class_pro/ibnb'
    
    batch_list = ['batch_1','batch_2','batch_3','batch_4']
    data_path_list = ['/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_1_SYSUCC',
                      '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_2_SYSUCC',
                      '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_3_SYSUCC',
                      '/cptjack/totem_data_backup/totem/COLORECTAL_DATA/batch_4_TCGA_analysis']
    npy_list = [os.path.join(npy_sub_dir,'batch_1_SYSUCC'),
                os.path.join(npy_sub_dir,'batch_2_SYSUCC'),
                os.path.join(npy_sub_dir,'batch_3_SYSUCC'),
                os.path.join(npy_sub_dir,'batch_4_TCGA_analysis')]
    
    etl_process(target_file,batch_list,data_path_list,npy_list,image_save_dir)