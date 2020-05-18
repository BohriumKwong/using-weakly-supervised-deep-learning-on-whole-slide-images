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
from Focal_Loss import focal_loss
import time
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm as tqdm
import pickle
import copy
from mark_result import result_excel_origin



def main(parser):
    global args, best_acc
    args = parser.parse_args()
    best_acc = 0

    #cnn
    model  = densenet121_ibn_b(num_classes=2,pretrained = False)
#    model = models.resnet34(pretrained = False)
#    model_path = model_path = '/your_dir/resnet34-333f7ec4.pth'
#    model_dict = torch.load('output/2020_03_06_CNN_checkpoint_best_3.9.pth')
#   如果加载自己模型就改为使用上述两句命令
#    model.fc = nn.Linear(model.fc.in_features, 2)
    model = nn.DataParallel(model.cuda())
#    model.load_state_dict(model_dict['state_dict'])
#    criterion = focal_loss(alpha=[1,args.weights/(1-args.weights)], gamma=2, num_classes = 2)
    
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    cudnn.benchmark = True

    #normalization
#    normalize = transforms.Normalize(mean=[0.736, 0.58, 0.701],std=[0.126, 0.144, 0.113])
    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
    val_trans = transforms.Compose([transforms.ToTensor()])
    
    with open(args.select_lib, 'rb') as fp:
        target_train_slide,target_val_slide = pickle.load(fp) 
    #load data
    train_dset = MILdataset(args.train_lib, args.k,train_trans,args.train_dir,target_train_slide,)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_dset = MILdataset(args.train_lib, 0,val_trans,args.train_dir,target_val_slide)
    val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    time_mark = time.strftime('%Y_%m_%d_',time.localtime(time.time()))
#    time_mark = '2020_03_06_'
    #以当前时间作为保存的文件名标识        
        
    #open output file
    fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_224_nonor_CE.csv'), 'w')
    fconv.write(' ,Train,,,,Validation,,,,Train_whole,,,Validation_whole,,\n')
    fconv.write('epoch,train_precision,train_recall,train_f1,train_loss,\
val_precision,val_recall,val_f1,val_loss,true_precision,true_recall,true_f1,true_precision,true_recall,true_f1')
    fconv.close()
    topk_list = []
    #用于存储每一轮算出来的top k index
    early_stop_count = 0
    #标记是否early stop的变量，该变量>epochs*2/3时,就开始进行停止训练的判断
    list_save_dir = os.path.join('output','topk_list','minmax')
    if not os.path.isdir(list_save_dir): os.makedirs(list_save_dir)
    
    best_metric_probs_inf_save = {'train_dset_slideIDX':train_dset.slideIDX,
                                  'train_dset_grid':train_dset.grid,
                                  'val_dset_slideIDX':val_dset.slideIDX,
                                  'val_dset_grid':val_dset.grid
                                  }
    #该字典主要用于保存最佳模型对应的train_probs和val_probs,以便用于后续的test和特征提取。
    # 之所以还要保存对应的dset_slideIDX和dset_grid是因为当train_dset出现一些slide的gird比top k的k数值还少时,\
    # 这部分的slide就会进行随机重复采样,下次直接调用的时候难免会出现部分gird和probs的记录不一致的情况,为确保严谨,\
    # 需要将上述这些列表也保存下来。然而val_dset默认不会出现这种情况,在这里也进行保存只是为了信息的一致性。
    eopch_save = {}
    #loop throuh epochs
    for epoch in range(args.nepochs):
        if epoch >=args.nepochs*2/3 and early_stop_count >= 3:
            print('Early stop at Epoch:'+ str(epoch+1))
            break
        start_time = time.time()
        #Train
        topk_exist_flag = False
        if os.path.exists(os.path.join(list_save_dir, time_mark + '_224.pkl')) and epoch ==0:
            with open(os.path.join(list_save_dir, time_mark + '_224.pkl'), 'rb') as fp:
                topk_list = pickle.load(fp)
                
#            topk = topk_list[-1][0]
            train_probs = topk_list[-1][1]
            topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[train_dset.label_mark], args.k)
            topk_exist_flag = True

        else:
            train_dset.setmode(1)
            train_probs = inference(epoch, train_loader, model, args.batch_size, 'train')           
            topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[train_dset.label_mark], args.k)
#            t_pred = group_max(train_probs,topk,args.k)
        repeat = 2
        if epoch >= 2/3*args.nepochs :
#            repeat = np.random.choice([3,5])         
            #前10轮设定在训练时复制采样,后10轮后随机决定是否复制采样
            topk_last = topk_list[-1][0]
            if sum(np.not_equal(topk_last, topk)) < 0.01 * len(topk):
                early_stop_count +=1        
        if not topk_exist_flag:
            topk_list.append((topk.copy(),train_probs.copy()))
        with open(os.path.join(list_save_dir, time_mark + '_224.pkl'), 'wb') as fp:
            pickle.dump(topk_list, fp)           
        
        train_dset.maketraindata(topk,repeat)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        train_whole_precision,train_whole_recall,train_whole_f1,train_whole_loss = train_predict(epoch, train_loader, model, criterion, optimizer)
        print('\tTraining  Epoch: [{}/{}] Precision: {} Recall:{} F1score:{} Loss: {}'.format(epoch+1, \
              args.nepochs, train_whole_precision,train_whole_recall,train_whole_f1,train_whole_loss))
        
#        topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[train_dset.label_mark], args.k)
#        t_pred = group_max(train_probs,topk,args.k)
        t_pred = group_identify(train_dset.slideIDX,train_probs)
        metrics_meters = calc_accuracy(t_pred, train_dset.targets)
        result = '\n'+str(epoch+1) + ',' + str(train_whole_precision) + ',' + str(train_whole_recall) + ',' + str(train_whole_f1) + ',' + str(train_whole_loss) \
                + ',' + str(metrics_meters['precision']) + ',' + str(metrics_meters['recall']) + ','\
                + str(metrics_meters['f1score'])

        val_dset.setmode(1)
        val_whole_precision,val_whole_recall,val_whole_f1,val_whole_loss,val_probs = train_predict(epoch, val_loader, model, criterion, optimizer, 'val')
#        v_topk = group_argtopk(np.array(val_dset.slideIDX), val_probs[val_dset.label_mark], args.k)
#        v_pred = group_max(val_probs,v_topk,args.k)
        v_pred = group_identify(val_dset.slideIDX,val_probs)

        metrics_meters = calc_accuracy(v_pred, val_dset.targets)
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
        s = ', '.join(str_logs)
        print('\tValidation  Epoch: [{}/{}]  '.format(epoch+1, args.nepochs) + s)
        result = result + ',' + str(val_whole_precision) + ',' +str(val_whole_recall) + ',' +str(val_whole_f1) + ',' \
                 + str(val_whole_loss) + ','+ str(metrics_meters['precision']) + ',' + str(metrics_meters['recall']) + ','\
                 + str(metrics_meters['f1score'])
        fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_224_nonor_CE.csv'), 'a')
        fconv.write(result)
        fconv.close()
        #Save best model
        tmp_acc = val_whole_f1 #(metrics_meters['acc'] + metrics_meters['recall'])/2 #- metrics_meters['fnr']*args.weights
        if tmp_acc >= best_acc:
            best_acc = tmp_acc.copy()
#            obj = {
#                'epoch': epoch+1,
#                'state_dict': model.state_dict(),
#                'best_acc': best_acc,
#                'optimizer' : optimizer.state_dict()
#            }
#            torch.save(obj, os.path.join(args.output, time_mark +'CNN_checkpoint_best.pth'))
            best_metric_probs_inf_save['train_probs'] = train_probs.copy()
            best_metric_probs_inf_save['val_probs'] = val_probs.copy()
            
        if epoch > 0:
            eopch_save.update({epoch+1:copy.copy(model.state_dict())})
            result_excel_origin(train_dset,t_pred,time_mark + 'train_' + str(epoch+1))
            result_excel_origin(val_dset,v_pred,time_mark + 'val_'+ str(epoch+1))
#                np.save('output/numpy_save/' +time_mark + 'train_infer_probs_' + str(epoch+1) + '.npy',train_probs)
#                np.save('output/numpy_save/' +time_mark + 'val_infer_probs_' + str(epoch+1) + '.npy',val_probs)

                
        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch + 1,time.time() - start_time))                
#    with open(os.path.join(list_save_dir, time_mark + '.pkl'), 'wb') as fp:
#        pickle.dump(topk_list, fp)
    
        torch.save(best_metric_probs_inf_save, 'output/numpy_save/final/minmax/best_metric_probs_inf_224.db')
        torch.save(eopch_save, os.path.join(args.output, time_mark +'densenet121_ibn_b_checkpoint_224.pth'))
    

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

def train_predict(run, loader, model, criterion, optimizer,phase='train'):
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
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                else:
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

def group_argtopk(groups, data,k=1):
    #该方法用于新一轮训练前进行所有图像全图region块预测,提取指定的top k的index
    # 假设group(slideIDX)列表为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],以top k=2为例
    # 对应预测label=1的概率为np.array([0,0.32,0.45,0.56,0.1,0.23,0.47,0.55,0.76,0.98,0.99,0.2,0.39,0.6], dtype=np.float64)
    # 通过np.lexsort((data, groups))计算的order为np.array([3,2,0,1,7,8,9,6,10,5,4,11,13,12])
    # 根据下面的index计算过程,最终得出index为np.array([False, False,  True,  True, False, False, False, False, False,\
    #    True,  True, False,  True,  True], dtype=bool),最终得出的结果如下:
    # np.array([ 0,  1,  5,  4, 13, 12], dtype=np.int64),即选出结果概率数组中第3(0.56),2(0.45),8(0.99),7(0.98),12(0.39),13(0.6)
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

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
        

def group_max(probs,topk,k):
    #该方法计算过程大致和group_argtopk类似，不过指定top k为1,于Validation过程使用
    # 同样以上述的group(slideIDX)和预测的概率array为例,最终计算到的结果是array([0.56,0.99,0.6], dtype=float64)
    # 即每个WSI中,选择所有region中预测label=1的概率最高值作为当前WSI预测label=1的概率
#    out = np.empty(nmax)
#    out[:] = np.nan
##    whole_mean_proba = np.zeros((1,2))
#    order = np.lexsort((data, groups))
#    groups = groups[order]
#    data = data[order]
#    index = np.empty(len(groups), 'bool')
#    index[-1] = True
#    index[:-1] = groups[1:] != groups[:-1]
#    out[groups[index]] = data[index]
    #######################################
    #上述说明仅针对原来的group_max方法,下面重新编写的validation评估的方法,对于每一张图片来说, \
    # 现在预测的方法是基于每个slide的top k截图的概率相加,然后取最大值者作为该slide最终的预测标签
    select_probs = np.array([probs[x,:] for x in topk])
    predict_result = []

    for j in range(0,select_probs.shape[0],k):
        if np.sum(np.argmax(select_probs[j:j+k,:],axis=1)) >= k/2 :
            predict_result.append(1)
        else:
            predict_result.append(0)

    return np.array(predict_result)

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', k=0, transform=None,image_save_dir='',select_list = None):
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
        #slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i,g in enumerate(grid_list):
            if len(g) < k:
                g = g + [(g[x]) for x in np.random.choice(range(len(g)), k-len(g))]
            #当前slide已有的grid数量在k之下时,就进行随机重复采样            
            grid.extend(g)    
            slideIDX.extend([i]*len(g))
            if int(lib['targets'][i]) == 0:
                label_mark.extend([(True,False)]*len(g))
            elif int(lib['targets'][i]) == 1:
                label_mark.extend([(False,True)]*len(g))
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
    parser.add_argument('--train_lib', type=str, default='output/lib/512/tum_region_mid/cnn_train_data_lib.db', help='path to train MIL library binary')
    parser.add_argument('--select_lib', type=str, default='output/lib/512/tum_region_mid/spilt_train_val.db', help='path to validation MIL library binary. If present.')
    parser.add_argument('--train_dir',type=str, default='/cptjack/totem_disk/totem/colon_pathology_data/MIL_202005/224')
    parser.add_argument('--output', type=str, default='output/p_r_f1/minmax', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    # 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
    #parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.82, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=140, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    #parser.add_argument('--tqdm_visible',default = True, type=bool,help='keep the processing of tqdm visible or not, default: True')

    main(parser)
