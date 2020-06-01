# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import argparse
import random
#import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from densenet_ibn_b import densenet121_ibn_b
#import torchvision.models as models
#from Focal_Loss import focal_loss
import time
from sklearn.metrics import precision_recall_fscore_support,roc_curve,auc
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import pickle
#import copy
from mark_result import result_excel_origin,group_log_excel



def main(parser):
    global args, best_acc
    args = parser.parse_args()
    best_acc = 0

    #cnn
    model  = densenet121_ibn_b(num_classes=2,pretrained = False)
#    model = models.resnet18(pretrained = False)
#    model_path = model_path = '/your_dir/resnet34-333f7ec4.pth'
    model_dict = torch.load('output/2020_05_13_densenet121_ibn_b_checkpoint_224.pth')
#   如果加载自己模型就改为使用上述两句命令
#    model.fc = nn.Linear(model.fc.in_features, 2)
    model = nn.DataParallel(model.cuda())
    model.load_state_dict(model_dict[20])
#    criterion = focal_loss(alpha=[1,args.weights/(1-args.weights)], gamma=2, num_classes = 2)
    
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1, args.weights/(1-args.weights)])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    cudnn.benchmark = True

#    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor()])
    val_trans = transforms.Compose([transforms.ToTensor()])
    
#    with open(args.select_lib, 'rb') as fp:
#        target_train_slide,target_val_slide = pickle.load(fp) 
#    #load data
#    train_dset = MILdataset(args.train_lib, args.k,train_trans,args.train_dir,target_train_slide,)
#    train_loader = torch.utils.data.DataLoader(
#        train_dset,
#        batch_size=args.batch_size, shuffle=False,
#        num_workers=args.workers, pin_memory=False)
    start_time = time.time()
    batch_select = ['batch_3_SYSUCC','batch_5_SYSUCC']
    
    for i in range(len(batch_select)):
        val_dset = MILdataset(args.train_lib,val_trans,args.train_dir,[batch_select[i]])
        val_loader = torch.utils.data.DataLoader(
                val_dset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
     
        
        val_dset.setmode(1)
        val_whole_precision,val_whole_recall,val_whole_f1,val_whole_loss,val_probs = train_predict(0, val_loader, model, criterion, optimizer, 'val')
    #        v_topk = group_argtopk(np.array(val_dset.slideIDX), val_probs[val_dset.label_mark], args.k)
    #        v_pred = group_max(val_probs,v_topk,args.k)
    #    val_probs = np.load('output/numpy_save/' + batch_select[0] + '_val' + '.npy')
        v_pred = group_identify(val_dset.slideIDX,val_probs)
    
        metrics_meters = calc_accuracy(v_pred, val_dset.targets)
        fconv = open(os.path.join(args.output,batch_select[i] + '_metric.csv'), 'w')
        fconv.write('sample_precision,sample_recall,sample_f1,slide_precision,slide_recall,slide_f1')
        result = '\n' + str(val_whole_precision) + ',' +str(val_whole_recall) + ',' +str(val_whole_f1) + ',' \
                     + str(metrics_meters['precision']) + ',' + str(metrics_meters['recall']) + ','\
                     + str(metrics_meters['f1score'])
        fconv.write(result)
        fconv.close()
        
        result_excel_origin(val_dset,v_pred,batch_select[i] + '_val')
        np.save('output/numpy_save/' + batch_select[i] + '_val' + '.npy',val_probs)
    #                np.save('output/numpy_save/' +time_mark + 'val_infer_probs_' + str(epoch+1) + '.npy',val_probs)
    
        msi_pro = group_proba(val_dset.slideIDX, val_probs,0.5)
        fpr, tpr, thresholds = roc_curve(val_dset.targets,msi_pro,pos_label=1)
        roc_auc = auc(fpr, tpr)
    
        ## 绘制roc曲线图
        plt.subplots(figsize=(7,5.5));
        plt.plot(fpr, tpr, color='darkorange',linewidth=2, label='ROC curve (area = %0.3f)' % roc_auc);
        plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--');
        plt.xlim([0.0, 1.0]);
        plt.ylim([0.0, 1.05]);
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title(batch_select[i] + ' ROC Curve');
        plt.legend(loc="lower right");
    #    plt.show()
        plt.savefig(os.path.join('output','0.5',f"{batch_select[i]}_ROC.png"))
        
        group_log(val_dset.slidenames,val_dset.slideIDX,val_dset.targets,val_dset.label_mark,val_probs,batch_select[0]+'_metric_info')
        
        print(batch_select[i] + '\t has been finished, needed %.2f sec.' % (time.time() - start_time)) 

def inference(run, loader, model, batch_size,phase):
    model.eval()
    probs = np.zeros((1,2))
#    logs = {}
    whole_probably = 0.

    with torch.no_grad():
        with tqdm(loader, desc = 'Epoch:' + str(run+1) + ' ' + phase + '\'s inferencing', \
                  file=sys.stdout, disable = False) as iterator:
            for i, (input, _) in enumerate(iterator):
#                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(iterator)))
                input = input.cuda()
                output = F.softmax(model(input), dim=1)
                prob = output.detach().clone()
                prob = prob.cpu().numpy()                
                batch_proba = np.mean(prob,axis=0)
                probs = np.row_stack((probs,prob))
                whole_probably = whole_probably + batch_proba

                iterator.set_postfix_str('batch proba :' + str(batch_proba))                                    
                
            whole_probably = whole_probably / (i+1)
            iterator.set_postfix_str('Whole average probably is ' + str(whole_probably))
            
    probs = np.delete(probs, 0, axis=0)
    return probs.reshape(-1,2)

def train_predict(run, loader, model, criterion, optimizer,phase='train',grad_add = False,accumulation = 2):
    if phase == 'val':
        model.eval()
        probs = np.zeros((1,2))
    elif phase == 'train':
        model.train()
    whole_loss = 0.
    whole_acc = 0.
    whole_recall = 0.
    whole_f1 = 0.
    logs = {}

    with tqdm(loader, desc = 'Epoch:' + str(run+1) + ' is ' + phase, \
                  file=sys.stdout, disable= False) as iterator:
        with torch.set_grad_enabled(phase == 'train'):
            for i, (input, target) in enumerate(iterator):
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                _, pred = torch.max(output, 1)
                pred = pred.data.cpu().numpy()
                metrics_meters = calc_accuracy(pred, target.cpu().numpy())
                logs.update(metrics_meters)
                logs.update({'loss':loss.item()})
                if phase == 'train':
                    loss.backward()
                    if grad_add:
                        if (i+1)%accumulation == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    output = F.softmax(output, dim=1)
                    prob = output.detach().clone()
                    prob = prob.cpu().numpy()
                    probs = np.row_stack((probs,prob))
                    
                str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                s = ', '.join(str_logs)
                iterator.set_postfix_str(s)
                
                whole_acc += metrics_meters['precision']
                whole_recall += metrics_meters['recall']
                whole_f1 += metrics_meters['f1score']
                whole_loss += loss.item()
    
    if phase == 'train':
        return round(whole_acc/(i+1),3),round(whole_recall/(i+1),3),round(whole_f1/(i+1),3),round(whole_loss/(i+1),3)
    else:
        probs = np.delete(probs, 0, axis=0)
        return round(whole_acc/(i+1),3),round(whole_recall/(i+1),3),round(whole_f1/(i+1),3),round(whole_loss/(i+1),3),probs.reshape(-1,2)
    
def calc_accuracy(pred,real):
    if str(type(pred)) !="<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) !="<class 'numpy.ndarray'>":
        real = np.array(real)
#    neq = np.not_equal(pred, real)
#    err = float(neq.sum())/pred.shape[0]
#    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
#    fnr = np.logical_and(pred==0,neq).sum()/(real==1).sum() if (real==1).sum() >0 else 0.0
#    balanced_acc = balanced_accuracy_score(real,pred)
#    recall = recall_score(real,pred,average='weighted')
    precision,recall,fbeta_score,_=precision_recall_fscore_support(real,pred,average='binary')
    metrics_meters = {'precision': round(precision,3),'recall':round(recall,3),'f1score':round(fbeta_score,3)}  
    
    return metrics_meters


def group_identify(groups, probs):
    #group(slideIDX)
    #这个方法有别于下面的基于top k进行这个slide预测的方法(group_max)
    # 这次不再局限于top k而是对slide所有sample都进行统计 by Bohrium.Kwong 2020.05.12
    predict_result = []
    for i in np.unique(groups):
        index = np.array(groups) == i
        select_probs = probs[index,:]
#        result = np.argmax(np.sum(select_probs,axis=0))
        result = 0 if np.sum(select_probs[:,0] > select_probs[:,1]) >= select_probs.shape[0]/2 else 1
        predict_result.append(result)
    return np.array(predict_result)  
    
def group_proba(groups, probs,thresholds = 0.5,pos_label = 1):
    #group(slideIDX)
    #这个方法有别于下面的基于top k进行这个slide预测的方法(group_max)
    # 这次不再局限于top k而是对slide所有sample都进行统计 by Bohrium.Kwong 2020.05.12
    predict_result = []
    for i in np.unique(groups):
        index = np.array(groups) == i
        select_probs = probs[index,:]
#        result = np.argmax(np.sum(select_probs,axis=0))
        result = np.sum(select_probs[:,pos_label] >= thresholds)/select_probs.shape[0] 
        predict_result.append(result)
    return np.array(predict_result)     

def group_log(slide_name_lst,slideIDX,slide_target,label_mark,probs,save_name):
    start_time = time.time()
    precision_list = []
    recall_list = []
    f1_list = []
    msi_count_list = []
    for i ,slide_name in enumerate(slide_name_lst):
        index = np.array(slideIDX) == i
        select_probs = probs[index,:]
        msi_count_list.append(np.sum(select_probs[:,1] >= 0.5)/select_probs.shape[0])
        metrics_meters = calc_accuracy(np.argmax(select_probs,axis=1),\
                                       np.argmax(np.array(label_mark)[index,:],axis=1))
        precision_list.append(metrics_meters['precision'])
        recall_list.append(metrics_meters['recall'])
        f1_list.append(metrics_meters['f1score'])
    
    result_dict = {'slide_name':slide_name_lst,'label':slide_target,'sample_precision':precision_list,\
                   'sample_recall':recall_list,'sample_f1score':f1_list,'MSI_count':msi_count_list}
    group_log_excel(result_dict,save_name)
    print(save_name + ' finished, needed %.2f sec.' % (time.time() - start_time)) 


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None,image_save_dir='',select_list = None):
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
                if lib['batch'][i] in select_list:
                    slides_list.append(slide)
                    grid_list.append(lib['grid'][i])
                    targets_list.append(lib['targets'][i])
                    batch_list.append(lib['batch'][i])
            
        grid = []
        slideIDX = []
        label_mark = []
        #slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i,g in enumerate(grid_list):
#            if len(g) < k:
#                g = g + [(g[x]) for x in np.random.choice(range(len(g)), k-len(g))]
            #当前slide已有的grid数量在k之下时,就进行随机重复采样            
            if int(lib['targets'][i]) == 0:
#                g = random.sample(g, int(len(g)/4))
                label_mark.extend([(True,False)]*len(g))
            elif int(lib['targets'][i]) == 1:
#                g = random.sample(g, int(len(g)/1.5))
                label_mark.extend([(False,True)]*len(g))
            #根据label返回对应的True/False索引,好让在inference过程中执行min_max抽取\
            # 如果slide的标签是0就抽取0类概率的top k，反之就抽取1类概率的top k. by Bohrium.Kwong 2020.05.12
            grid.extend(g)    
            slideIDX.extend([i]*len(g))

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
        self.batch = batch_list
        self.image_save_dir = image_save_dir
        
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs,repeat=0):
        #repeat这个参数用于是否对采样进行复制,如果进行复制,就会在下面的_getitem_方法中对重复的样本进行不一样的颜色增强
        if abs(repeat) == 0:
            #repeat等于0的时,按用原来的方法进行生成筛选的数据,并不会进行h通道的颜色变换。
            self.t_data = [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
        else:
            repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1
            #通过该操作确保非奇数的repeat传参也能变为奇数  
            self.t_data = [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
            for y in range(-100,int(100 + repeat/2),int(100*2/repeat)):
                #将会在(-0,1,0.1)范围内按照repeat的数值进行区间划分(这也是要求repeat值必须为奇数的原因所在)
                # 通过上面的划分,可以确保除0外在(-0,1,0.1)都会划分为repeat-1倍,需要注意最后y的值必须控制在0.1以内
                self.t_data = self.t_data + [(self.batch[self.slideIDX[x]],self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],y/1000) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            # mode =1 为预测时使用，会从所有WSI文件中返回全部的region的图像
            slideIDX = self.slideIDX[index]
            (k,j) = self.grid[index]
#            img = self.slides[slideIDX].read_region(coord,self.level,(self.patch_size[slideIDX],\
#                                                    self.patch_size[slideIDX])).convert('RGB')
            target = self.targets[slideIDX]
            img = Image.open(os.path.join(self.image_save_dir,self.batch[slideIDX],str(target),self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j)+'.jpg'))
#            if img.size != (224,224):
#                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img,target
        
        elif self.mode == 2:
            # mode =2 为训练时使用，只会根据指定的index(经过上一轮MIL过程得出) \
            #   从全部WSI文件中筛选对应的坐标列表,返回相应的训练图像和label
            batch,slideIDX, (k,j), target,h_value = self.t_data[index]
            img = Image.open(os.path.join(self.image_save_dir,batch,str(self.targets[slideIDX]),self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j)+'.jpg'))

            if h_value > 0:
                hue_factor = random.uniform(h_value,0.1)
            elif h_value == 0:
                hue_factor = random.uniform(0,0)               
            elif h_value < 0:                
                hue_factor = random.uniform(-0.1,h_value)   
            img = functional.adjust_hue(img,hue_factor)
            # 只有在训练模式下才进行H通道变换的颜色增强方法
            # 如果在maketraindata方法设置采样复制,那么就会针对h_value的值进行不同方向的hue_factor生成,\
            #    从而达到复制的样本和原来的样本有不一样的增强的效果
#            if img.size != (224,224):
#                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
                   
    def __len__(self):
        if self.mode in [1,3]:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    parser.add_argument('--train_lib', type=str, default='output/lib/tum_region_mid/cnn_train_data_lib.db', help='path to train MIL library binary')
    parser.add_argument('--select_lib', type=str, default='output/lib/tum_region_mid/spilt_train_val.db', help='path to validation MIL library binary. If present.')
    parser.add_argument('--train_dir',type=str, default='/media/totem_disk/totem/MIL_202005/224', help='root path to whole dataset. If present.')
    parser.add_argument('--output', type=str, default='output', help='name of output dir')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    # 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
    #parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.55, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=100, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    #parser.add_argument('--tqdm_visible',default = True, type=bool,help='keep the processing of tqdm visible or not, default: True')

    main(parser)
