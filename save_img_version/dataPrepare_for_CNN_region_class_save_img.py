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


def etl_process(target_file,batch_list,data_path_list,npy_list,TCGA_USELESS,image_save_dir):
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
    
    
    if not os.path.exists('./output/lib/512/tum_region/cnn_train_data_lib.db'):
        # 最终保存全部数据的字典
        train_data_lib = {}
        train_slides_list = []   # 存储文件路径
        train_targets_list = []  # 存储目标信息
        train_grids_list = []    # 存储格点信息
        train_patch_size_list = [] # 存储WSI文件对应切片尺寸
        train_flag = 0           # 记录已采集slide的数量
        train_sum = 0            # 记录已采集的总切图数

    else:
        with open('./output/lib/512/tum_region/cnn_train_data_lib.db', 'rb') as fp:
           train_data_lib = pickle.load(fp)
           train_slides_list = train_data_lib['slides']
           train_flag = len(train_slides_list)
           train_grids_list = train_data_lib['grid']
           train_sum = np.sum([len(x) for x in train_grids_list])
           train_targets_list = train_data_lib['targets']
           train_patch_size_list = train_data_lib['patch_size'] 
    
    if not os.path.exists('./output/lib/512/tum_region/cnn_val_data_lib.db'):    
        val_data_lib = {}
        val_slides_list = []   # 存储文件路径
        val_targets_list = []  # 存储目标信息
        val_grids_list = []    # 存储格点信息
        val_patch_size_list = [] # 存储WSI文件对应切片尺寸
        val_flag = 0 # 记录已采集slide的数量
        val_sum = 0 # 记录已采集的总切图数
    else:
        with open('./output/lib/512/tum_region/cnn_val_data_lib.db', 'rb') as fp:
           val_data_lib = pickle.load(fp)
           val_slides_list = val_data_lib['slides']
           val_flag = len(val_slides_list)
           val_grids_list = val_data_lib['grid']
           val_sum = np.sum([len(x) for x in val_grids_list])
           val_targets_list = val_data_lib['targets']
           val_patch_size_list = val_data_lib['patch_size'] 
    # 如果已有有效的lib.db文件,则相关list直接读入文件而不是重零开始计算    

    def save_lib():            
        train_data_lib['slides'] = train_slides_list
        train_data_lib['grid'] = train_grids_list
        train_data_lib['targets'] = train_targets_list
        train_data_lib['level'] = level
        train_data_lib['patch_size'] = train_patch_size_list
        if not os.path.isdir('./output/lib/512/tum_region'): os.makedirs('./output/lib/512/tum_region')
#        torch.save(train_data_lib, './output/lib/512/all_region/sample_pro_' + str(patch_widen_value) + '_cnn_train_data_lib.db')
        with open('./output/lib/512/tum_region/cnn_train_data_lib.db', 'wb') as fp:
            pickle.dump(train_data_lib, fp)

        
        val_data_lib['slides'] = val_slides_list
        val_data_lib['grid'] = val_grids_list
        val_data_lib['targets'] = val_targets_list
        val_data_lib['level'] = level
        val_data_lib['patch_size'] = val_patch_size_list
#        torch.save(val_data_lib, 'output/lib/512/all_region/sample_pro_' + str(patch_widen_value) + '_cnn_val_data_lib.db')
        with open('./output/lib/512/tum_region/cnn_val_data_lib.db', 'wb') as fp:
            pickle.dump(val_data_lib, fp)  
        
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
            if i ==3:
                npypath = npypath + glob.glob(os.path.join(TCGA_USELESS, "*" + basename + "*output.npy"))
                #在当前场景中,第四批TCGA的slide分为正常和非正常(TCGA_USELESS代表非正常)两种情况分开存储,所以处理第四批的时候要合并处理
                
            if len(wispath)==1 and len(npypath) ==1 and basename not in train_slides_list + val_slides_list:
                data_df.iloc[index,3] = wispath[0]
                data_df.iloc[index,4] = npypath[0]
    
            # ---------------------- 开始处理数据，获取 lib ---------------------- #
                start_time = time.time()
                slide = Slide(wispath[0])
                try: 
                    if round(slide.get_mpp()/0.00025) == 1:
                        patch_size = 1024#1024
                    elif round(slide.get_mpp()/0.00025) == 2:
                        patch_size = 512#512
                    else:
                        patch_size = 256#256
                except Exception:
                    patch_size = 512
                    
                thumb = np.load(npypath[0])
                if len(thumb.shape)==3:
                    if thumb.shape[2]>1:
                        thumb = np.argmax(thumb,axis=2)
                        thumb = np.uint8(thumb)
                # 如果加载的分类结果矩阵是维度>2且通道数>1,说明这个矩阵是N分类概率矩阵,需要进行argmax处理还原为常用的分类结果矩阵
                try:
                    svs_level = 2 if len(slide.level_dimensions) >2 else len(slide.level_downsamples) -1
                    size = slide.get_level_dimension(svs_level)

                    if size[1] - thumb.shape[0] + size[0] - thumb.shape[1] > 200:
                       thumb =  matrix_resize(slide,thumb,patch_size,svs_level)
#                    thumb = cv2.dilate(np.uint8(thumb==8), kernel, iterations=1)    
                    w, h = slide.get_level_dimension(0)
                    level_downsamples = round(slide.get_level_downsample(svs_level))
        #            sample_proba = min(1200 * (patch_size / level_downsamples)**2 / np.sum(thumb),1)
                #        if random.randint(0, 21) < 14:
                #    if np.random.binomial(1,0.72):
                    if batch != 'batch_4':               
                        cur_patch_cords = []
                
                        for j in range(0, h, patch_size):
                            for k in range(0, w, patch_size):
                                bottom = int(j / level_downsamples)
                                top = bottom + int(patch_size / level_downsamples) -1
                                left = int(k / level_downsamples)
                                right = left + int(patch_size / level_downsamples) -1
                                #这里以区域分类结果作为过滤条件,8即TUM区域
                                if np.sum(thumb[bottom : top,left : right ] == 8) > 0.7 * (patch_size / level_downsamples)**2:
        #                        and np.random.binomial(1,sample_proba):
                                    img = slide.read_region((k,j),0,(patch_size,patch_size)).convert('RGB')
                                    img = img.resize((224,224),Image.BILINEAR)
                                    img.save(os.path.join(image_save_dir,'train',str(data_df.iloc[index,1]),\
                                                        basename + '_' + str(k) + '_' + str(j)+'.jpg'))
                                    cur_patch_cords.append((k,j))
                        data_df.iloc[index,6] = len(cur_patch_cords)
                        if len(cur_patch_cords) >0:
                            train_flag = train_flag + 1                                
                            train_slides_list.append(basename)
                            train_patch_size_list.append(patch_size)
                            train_targets_list.append(data_df.iloc[index,1])
                            train_grids_list.append(cur_patch_cords)
                            train_sum = train_sum + len(cur_patch_cords)
        
                    else:            
                        cur_patch_cords = []
                
                        for j in range(0, h, patch_size):
                            for k in range(0, w, patch_size):
                                bottom = int(j / level_downsamples)
                                top = bottom + int(patch_size / level_downsamples) -1
                                left = int(k / level_downsamples)
                                right = left + int(patch_size / level_downsamples) -1
                                # 根据当前循环移动到的patch所在的区域，在上述提取的背景/组织区域二值图中映射同样的位置,当覆盖的像素点的占比超过64%时,、
                                #   说明该patch所在是我们感兴趣的区域,可以进行patch坐标记录
                                if np.sum(thumb[bottom : top,left : right ] == 8) > 0.7 * (patch_size / level_downsamples)**2:
        #                        and np.random.binomial(1,sample_proba):
                                    img = slide.read_region((k,j),0,(patch_size,patch_size)).convert('RGB')
                                    img = img.resize((224,224),Image.BILINEAR)
                                    img.save(os.path.join(image_save_dir,'val',str(data_df.iloc[index,1]),\
                                                        basename + '_' + str(k) + '_' + str(j)+'.jpg'))
        
                                    cur_patch_cords.append((k,j))
                        data_df.iloc[index,6] = len(cur_patch_cords)
                        if len(cur_patch_cords) >0 :
                            val_flag = val_flag + 1
                            val_slides_list.append(basename)
                            val_patch_size_list.append(patch_size)
                            val_targets_list.append(data_df.iloc[index,1])  
                            val_grids_list.append(cur_patch_cords)
                            val_sum = val_sum + len(cur_patch_cords)
        
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
        whole_data.to_excel('tum_etl_tag_512.xlsx',index = False)
        
    print('%d WSI files ,total %d samples in train data set.' %(train_flag,train_sum))
    print('%d WSI files ,total %d samples in val data set.' %(val_flag,val_sum))
        

if __name__ == '__main__':
    image_save_dir = '/cptjack/totem_other/totem/weakly_supervised_MIL_202005'
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
    TCGA_USELESS = os.path.join(npy_sub_dir,'batch_4_TCGA_useless')
    
    etl_process(target_file,batch_list,data_path_list,npy_list,TCGA_USELESS,image_save_dir)