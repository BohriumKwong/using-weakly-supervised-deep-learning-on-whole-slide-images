# -*- coding: utf-8 -*-
# @Time    : 2020.6.17
# @Author  : Bohrium.Kwong
# @Licence : bio-totem

import sys
import os
import numpy as np
import argparse
import random
#import openslide
import PIL.Image as Image
from skimage import io
from PIL.ImageFilter import GaussianBlur
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.backends.cudnn as cudnn
#import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
#import torchvision.models as models
#import time
#from sklearn.metrics import precision_recall_fscore_support
#from tqdm import tqdm as tqdm
import pickle
#import copy


def cut_img(img,size = 224,loc = 'LT',is_random=False):
    # 该方法是选定指定部位[左上LT,右上RT,左下LB,右下RB]对传入大小为(512,512)的图像进行裁截的方法
    # 其中is_random为True时进行随机定位,为False时以中心定位进行裁截
    if size == 224:
        if is_random:
            if loc == 'LT':
                grid = [random.randint(0, 32),random.randint(0, 32)]
            elif loc == 'RT':
                grid = [random.randint(256, 288), random.randint(0, 32)]
            elif loc == 'LB':
                grid = [random.randint(0, 32), random.randint(256, 288)]
            elif loc == 'RB':
                grid = [random.randint(256, 288), random.randint(256, 288)]
        else:
            if loc == 'LT':
                grid = [32, 32]
            elif loc == 'RT':
                grid = [288, 32]
            elif loc == 'LB':
                grid = [32, 288]
            elif loc == 'RB':
                grid = [288, 288]
    elif size == 448:
        if is_random:
            grid = [random.randint(0, 64),random.randint(0, 64)]
        else:
            grid = [32, 32]
        
    return img[grid[1]:grid[1] + size,grid[0]:grid[0]+ size,:]



class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', size = 224,transform=None,image_save_dir='',select_list = None):
#        lib = torch.load(libraryfile)
#        slides = lib['slides']
#        patch_size = lib['patch_size']
#        for i,name in enumerate(lib['slides']):
#            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
##            sys.stdout.flush()
##            slides.append(openslide.OpenSlide(name))
#            patch_size.append(int(lib['patch_size'][i]))
            #获取WSI文件对应的切片大小,因为在生成lib时,已经确保lib['slides']和lib['patch_size']顺序是对应的,\
            # 所以可以在一个循环中使用相同的index进行定位
        with open(libraryfile, 'rb') as fp:
            lib = pickle.load(fp)       
        print('')
        #Flatten grid
        
        if select_list is None:
            slides_list = lib['slides']
            grid_list = lib['grid']
            targets_list = lib['targets']
            batch_list = lib['batch']
        else:
            slides_list = []
            grid_list = []
            targets_list = []
            batch_list = []
            for i,slide in enumerate(lib['slides']):
                if slide in select_list:
                    slides_list.append(slide)
                    grid_list.append(lib['grid'][i])
                    targets_list.append(lib['targets'][i])
                    batch_list.append(lib['batch'][i])
            
        grid = []
        slideIDX = []
        label_mark = []
        loc = []
        loc_flag = ['LT','RT','LB','RB']
        # loc存放的是每一个截图的坐标方向标记(分别是[左上LT,右上RT,左下LB,右下RB])
        #slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i,g in enumerate(grid_list):
#            if len(g) < k/4:
#                g = g + [(g[x]) for x in np.random.choice(range(len(g)), int(k/4-len(g)))]
            #当前slide已有的grid数量在k之下时,就进行随机重复采样
            for l in range(int(512/size)**2):
                #根据切图大小和448的关系决定重复采样的数量，如果是448/224就按照田字形采样4次
                loc.extend([loc_flag[l]]*len(g))
                grid.extend(g)
                
            slideIDX.extend([i]*len(g)*int(512/size)**2)
            if int(lib['targets'][i]) == 0:
                label_mark.extend([(True,False)]*len(g)*int(512/size)**2)
            elif int(lib['targets'][i]) == 1:
                label_mark.extend([(False,True)]*len(g)*int(512/size)**2)
            #根据label返回对应的True/False索引,好让在inference过程中执行min_max抽取\
            # 如果slide的标签是0就抽取0类概率的top k，反之就抽取1类概率的top k. by Bohrium.Kwong 2020.05.12


        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = slides_list
        self.targets = targets_list
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = 1
        self.label_mark = label_mark
#        self.patch_size = lib['patch_size']
#        self.level = lib['level']
        self.loc = loc
        self.size = size
        self.batch = batch_list
        self.image_save_dir = image_save_dir
        
    def setmode(self,mode):
        if mode in [1,2]:
            self.mode = mode
    def maketraindata(self, idxs = None,repeat=0):
        if idxs is None:
            lst = range(len(self.grid))
        else:
            lst = idxs
        #repeat这个参数用于是否对采样进行复制,如果进行复制,就会在下面的_getitem_方法中对重复的样本进行不一样的颜色增强
        #repeat等于0的时,按用原来的方法进行生成筛选的数据,并不会进行h通道的颜色变换。
        self.t_data = [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0,self.loc[x]) for x in lst]
#        self.t_data = self.t_data + [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0,'RT') for x in lst]
#        self.t_data = self.t_data + [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0,'LB') for x in lst]
#        self.t_data = self.t_data + [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0,'RB') for x in lst]
        if abs(repeat) != 0 and idxs is not None and self.mode==2:
            # 当且仅当传入idxs且repeat不为0时才进行数据集扩充
            repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1
            #通过该操作确保非奇数的repeat传参也能变为奇数  
            for y in range(-100,int(100 + repeat/2),int(100*2/repeat)):
                #将会在(-0,1,0.1)范围内按照repeat的数值进行区间划分(这也是要求repeat值必须为奇数的原因所在)
                # 通过上面的划分,可以确保除0外在(-0,1,0.1)都会划分为repeat-1倍,需要注意最后y的值必须控制在0.1以内
                self.t_data = self.t_data + [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],y/1000,random.choice['LT','RT','LB','RB']) for x in lst]
    def shuffletraindata(self):
        if self.mode == 2:
            self.t_data = random.sample(self.t_data, len(self.t_data))
    
    def __getitem__(self,index):
        if self.mode == 1:
            
            batch,slideIDX, (k,j), target,_,loc = self.t_data[index]
            if self.size in (448,224):
    #            img = Image.open(os.path.join(self.image_save_dir,batch,str(self.targets[slideIDX]),self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j)+'.jpg'))
                img = io.imread(os.path.join(self.image_save_dir,self.slidenames[slideIDX],self.slidenames[slideIDX]+'_('+str(k)+','+str(j)+').jpg'))
    #            io.imread
                img = cut_img(img,self.size,loc,False)
                img = Image.fromarray(img)           
                    
                if img.size != (224,224):
                    img = img.resize((224,224),Image.BILINEAR)
            else:
                img = Image.open(os.path.join(self.image_save_dir,self.slidenames[slideIDX],self.slidenames[slideIDX]+'_('+str(k)+','+str(j)+').jpg'))
#                img = img.resize((224,224),Image.BILINEAR)
            
            
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        
        elif self.mode == 2:
            # mode =2 为训练时使用，只会根据指定的index(经过上一轮MIL过程得出) \
            #   从全部WSI文件中筛选对应的坐标列表,返回相应的训练图像和label
            batch,slideIDX, (k,j), target,h_value,loc = self.t_data[index]
#            img = Image.open(os.path.join(self.image_save_dir,batch,str(self.targets[slideIDX]),self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j)+'.jpg'))
            if self.size <= 448:
                img = io.imread(os.path.join(self.image_save_dir,self.slidenames[slideIDX],self.slidenames[slideIDX]+'_('+str(k)+','+str(j)+').jpg'))
                img = cut_img(img,self.size,loc,True)
                img = Image.fromarray(img)
            
                if img.size != (224,224):
                    img = img.resize((224,224),Image.BILINEAR)
            else:
                img = Image.open(os.path.join(self.image_save_dir,self.slidenames[slideIDX],self.slidenames[slideIDX]+'_('+str(k)+','+str(j)+').jpg'))
#                img = img.resize((224,224),Image.BILINEAR)
            #对H通道的像素值进行线性变换实现基于H通道的数据增强
#            if h_value > 0:
#                hue_factor = random.uniform(h_value,0.1)
#            elif h_value == 0:
#                hue_factor = random.uniform(0,0)               
#            elif h_value < 0:                
#                hue_factor = random.uniform(-0.1,h_value)
#            #对原图进行一定程度的高斯模糊处理实现模糊的数据增强
            if np.random.binomial(1,0.5):
                radius = random.uniform(0.5,1.1)
                img = img.filter(GaussianBlur(radius))
#                    
#            if hue_factor !=0:
#                img = functional.adjust_hue(img,hue_factor)
                
            # 只有在训练模式下才进行H通道变换的颜色增强方法
            # 如果在maketraindata方法设置采样复制,那么就会针对h_value的值进行不同方向的hue_factor生成,\
            #    从而达到复制的样本和原来的样本有不一样的增强的效果
            
            if self.transform is not None:
                img = self.transform(img)
            return img, target
                   
    def __len__(self):
        return len(self.t_data)

def main(parser):
    global args, best_acc
    args = parser.parse_args()
    best_acc = 0



    #normalization
#    normalize = transforms.Normalize(mean=[0.736, 0.58, 0.701],std=[0.126, 0.144, 0.113])
    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
    val_trans = transforms.Compose([transforms.ToTensor()])
    
    with open(args.select_lib, 'rb') as fp:
        target_train_slide,target_val_slide = pickle.load(fp) 
    #load data
    train_dset = MILdataset(args.train_lib, args.k,224,train_trans,args.train_dir,target_train_slide)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_dset = MILdataset(args.train_lib, 0,224,val_trans,args.train_dir,target_val_slide)
    val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    # parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
    # parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
    # parser.add_argument('--output', type=str, default='.', help='name of output file')
    # parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
    # parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
    # parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    # parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    # parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    # parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument('--train_lib', type=str, default='/media/totem_disk/totem/kwong/NEW_MIL_CODE/output/lib/tum_region_mid/cnn_train_data_lib.db', help='path to train MIL library binary')
    parser.add_argument('--select_lib', type=str, default='/media/totem_disk/totem/kwong/NEW_MIL_CODE/output/lib/tum_region_mid/spilt_train_val_fold_1.db', help='path to validation MIL library binary. If present.')
    parser.add_argument('--train_dir',type=str, default='/media/totem_disk/totem/MIL_202005/224', help='root path to whole dataset. If present.')
    parser.add_argument('--output', type=str, default='output/3_FOLD/1', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    # 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
    #parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.80, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=100, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    #parser.add_argument('--tqdm_visible',default = True, type=bool,help='keep the processing of tqdm visible or not, default: True')

    main(parser)
