# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import torchvision.models as models
import time
from sklearn.metrics import balanced_accuracy_score,recall_score
from tqdm import tqdm as tqdm
import pickle
from mark_result import result_excel_origin



def main(parser):
    global args, best_acc
    args = parser.parse_args()
    best_acc = 0

    #cnn
    model = models.resnet34(pretrained = False)
#    model_path = model_path = '/your_dir/resnet34-333f7ec4.pth'
    model_dict = torch.load('output/2020_03_06_CNN_checkpoint_best_3.9.pth')
#   如果加载自己模型就改为使用上述两句命令
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(model_dict['state_dict'])

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights, args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), 
                                        normalize])
    val_trans = transforms.Compose([transforms.ToTensor(), normalize])
    
    #load data
    train_dset = MILdataset(args.train_lib, args.k,train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, 0,val_trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    time_mark = time.strftime('%Y_%m_%d_',time.localtime(time.time()))
#    time_mark = '2020_03_06_'
    #以当前时间作为保存的文件名标识        
        
    #open output file
    fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_224.csv'), 'w')
    fconv.write(' ,Training,,,,Train_whole,,,Validation,,\n')
    fconv.write('epoch,train_acc,train_recall,train_fnr,train_loss,true_acc,true_recall,true_fnr,acc,recall,fnr')
    fconv.close()
    topk_list = []
    #用于存储每一轮算出来的top k index
    early_stop_count = 0
    #标记是否early stop的变量，该变量>3时,就停止训练
    list_save_dir = os.path.join('output','topk_list')
    if not os.path.isdir(list_save_dir): os.makedirs(list_save_dir)
    
    best_metric_probs_inf_save = {'train_dset_slideIDX':train_dset.slideIDX,
                                  'train_dset_grid':train_dset.grid,
                                  'val_dset_slideIDX':val_dset.slideIDX,
                                  'val_dset_grid':val_dset.grid
                                  }
    #loop throuh epochs
    for epoch in range(args.nepochs):
        if epoch >=6 and early_stop_count >= 3:
            print('Early stop at Epoch:'+ str(epoch+1))
            break
        start_time = time.time()
        #Train
        topk_exist_flag = False
        if os.path.exists(os.path.join(list_save_dir, time_mark + '.pkl')) and epoch ==0:
            with open(os.path.join(list_save_dir, time_mark + '.pkl'), 'rb') as fp:
                topk,train_probs = pickle.load(fp)
                
            topk = topk[-1]
            topk_exist_flag = True

        else:
            train_dset.setmode(1)
            train_probs = inference(epoch, train_loader, model, args.batch_size, 'train')           
            topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:,1], args.k)
            t_pred = group_max(train_probs,topk,args.k)
        repeat = 5
        if epoch >= 2/3*args.nepochs :
            repeat = np.random.choice([3,5])         
            #前10轮设定在训练时复制采样,后10轮后随机决定是否复制采样
            topk_last = topk_list[-1][0]
            if sum(np.not_equal(topk_last, topk)) < 0.01 * len(topk):
                early_stop_count +=1        
        if not topk_exist_flag:
            topk_list.append((topk.copy(),train_probs.copy()))
        with open(os.path.join(list_save_dir, time_mark + '.pkl'), 'wb') as fp:
            pickle.dump(topk_list, fp)           
        
        train_dset.maketraindata(topk,repeat)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        whole_acc,whole_recall,whole_fnr,whole_loss = train(epoch, train_loader, model, criterion, optimizer)
        print('\tTraining  Epoch: [{}/{}] Acc: {} Recall:{} Fnr:{} Loss: {}'.format(epoch+1, \
              args.nepochs, whole_acc,whole_recall,whole_fnr,whole_loss))
        
        topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:,1], 1)
        t_pred = group_max(train_probs,topk,1)        
        metrics_meters = calc_accuracy(t_pred, train_dset.targets)
        result = '\n'+str(epoch+1) + ',' + str(whole_acc) + ',' +str(whole_recall)+ ',' +str(whole_fnr)+ ',' +str(whole_loss) \
                + ','+ str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ','\
                + str(metrics_meters['fnr'])

        #Validation
#        if args.val_lib and (epoch+1) % args.test_every == 0:
        val_dset.setmode(1)
        val_probs = inference(epoch, val_loader, model, args.batch_size, 'val')
        v_topk = group_argtopk(np.array(val_dset.slideIDX), val_probs[:,1], 1)
        v_pred = group_max(val_probs,v_topk,1)
#        np.save('output/numpy_save/val_infer_probs.npy',pred)
        metrics_meters = calc_accuracy(v_pred, val_dset.targets)
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
        s = ', '.join(str_logs)
        print('\tValidation  Epoch: [{}/{}]  '.format(epoch+1, args.nepochs) + s)
        result = result + ','+ str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ','\
                 + str(metrics_meters['fnr'])
        fconv = open(os.path.join(args.output, time_mark + 'CNN_convergence_224.csv'), 'a')
        fconv.write(result)
        fconv.close()
        #Save best model
        tmp_acc = (metrics_meters['acc'] + metrics_meters['recall'])/2 - metrics_meters['fnr']*args.weights
        if tmp_acc >= best_acc:
            best_acc = tmp_acc.copy()
            obj = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.output, time_mark +'CNN_checkpoint_best.pth'))
            
            if epoch > 0:
                result_excel_origin(train_dset,t_pred,time_mark + 'train_' + str(epoch+1))
                result_excel_origin(val_dset,v_pred,time_mark + 'val_'+ str(epoch+1))
#                np.save('output/numpy_save/' +time_mark + 'train_infer_probs_' + str(epoch+1) + '.npy',train_probs)
#                np.save('output/numpy_save/' +time_mark + 'val_infer_probs_' + str(epoch+1) + '.npy',val_probs)
                best_metric_probs_inf_save['train_probs'] = train_probs.copy()
                best_metric_probs_inf_save['val_probs'] = val_probs.copy()
                
        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch + 1,time.time() - start_time))                   
    with open(os.path.join(list_save_dir, time_mark + '.pkl'), 'wb') as fp:
        pickle.dump(topk_list, fp)
    
    torch.save(best_metric_probs_inf_save, 'output/numpy_save/final/best_metric_probs_inf.db')
    

def inference(run, loader, model, batch_size,phase):
    model.eval()
    probs = np.zeros((1,2))
#    logs = {}
    whole_probably = 0.

    with torch.no_grad():
        with tqdm(loader, desc = 'Epoch:' + str(run+1) + ' ' + phase + '\'s inferencing', \
                  file=sys.stdout, disable = False) as iterator:
            for i, input in enumerate(iterator):
#                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run+1, args.nepochs, i+1, len(iterator)))
                input = input.cuda()
                output = F.softmax(model(input), dim=1)
                prob = output.detach().clone()
                prob = prob.cpu().numpy()                
                batch_proba = np.mean(prob,axis=0)
                probs = np.row_stack((probs,prob))
                whole_probably = whole_probably + batch_proba
#                temp_log = {'batch proba ': str(batch_proba)}
#                logs.update(temp_log)
#                
#                str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
#                s = ', '.join(str_logs)
                iterator.set_postfix_str('batch proba :' + str(batch_proba))                                    
                
            whole_probably = whole_probably / (i+1)
            iterator.set_postfix_str('Whole average probably is ' + str(whole_probably))
            
    probs = np.delete(probs, 0, axis=0)
    return probs.reshape(-1,2)

def train(run, loader, model, criterion, optimizer):
    model.train()
    whole_loss = 0.
    whole_acc = 0.
    whole_recall = 0.
    whole_fnr = 0.
    logs = {}

    with tqdm(loader, desc = 'Epoch:' + str(run+1) + ' is trainng', \
                  file=sys.stdout, disable= False) as iterator:
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
            loss.backward()
            optimizer.step()
            
            str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
            s = ', '.join(str_logs)
            iterator.set_postfix_str(s)
            
            whole_acc += metrics_meters['acc']
            whole_recall += metrics_meters['recall']
            whole_fnr += metrics_meters['fnr']
            whole_loss += loss.item()
    return round(whole_acc/(i+1),3),round(whole_recall/(i+1),3),round(whole_fnr/(i+1),3),round(whole_loss/(i+1),3)

def calc_accuracy(pred,real):
    if str(type(pred)) !="<class 'numpy.ndarray'>":
        pred = np.array(pred)
    if str(type(real)) !="<class 'numpy.ndarray'>":
        real = np.array(real)
    neq = np.not_equal(pred, real)
#    err = float(neq.sum())/pred.shape[0]
#    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()    
    balanced_acc = balanced_accuracy_score(real,pred)
    recall = recall_score(real,pred,average='weighted')
    metrics_meters = {'acc': round(balanced_acc,3),'recall':round(recall,3),'fnr':round(fnr,3)}  
    
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
    # 预测的label取决于所有region截图分类概率相加后最大值者为准
    select_probs = np.array([probs[x,:] for x in topk])
    predict_result = []
#    groups = np.array(groups)
#    for i in np.unique(groups):
##        mean_proba = np.mean(data[groups==i],axis=0)
##        whole_mean_proba = np.row_stack((whole_mean_proba,mean_proba))
##        out[i] = np.argmax(mean_proba)
    for j in range(0,select_probs.shape[0],k):
        if np.sum(np.argmax(select_probs[j:j+k,:],axis=1)) >= k/2 :
            predict_result.append(1)
        else:
            predict_result.append(0)
            
#    return np.delete(whole_mean_proba, 0, axis=0)
    return np.array(predict_result)

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', k=0, transform=None):
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        for i,name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
            #获取WSI文件对应的切片大小,因为在生成lib时,已经确保lib['slides']和lib['patch_size']顺序是对应的,\
            # 所以可以在一个循环中使用相同的index进行定位
        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        #slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i,g in enumerate(lib['grid']):
            if len(g) < k:
                g = g + [(g[x]) for x in np.random.choice(range(len(g)), k-len(g))]               
            grid.extend(g)    
            slideIDX.extend([i]*len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.patch_size = patch_size
        self.level = lib['level']
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs,repeat=0):
        #repeat这个参数用于是否对采样进行复制,如果进行复制,就会在下面的_getitem_方法中对重复的样本进行不一样的颜色增强
        if abs(repeat) == 0:
            self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
        else:
            repeat = abs(repeat) if repeat % 2 == 1 else abs(repeat) + 1       
            self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],0) for x in idxs]
            for y in range(-100,int(100 + repeat/2),int(100*2/repeat)):
                self.t_data = self.t_data + [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]],y/1000) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            # mode =1 为预测时使用，会从所有WSI文件中返回全部的region的图像
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]

            img = self.slides[slideIDX].read_region(coord,self.level,(self.patch_size[slideIDX],\
                                                    self.patch_size[slideIDX])).convert('RGB')
            if img.size != (224,224):
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        
        elif self.mode == 2:
            # mode =2 为训练时使用，只会根据指定的index(经过上一轮MIL过程得出) \
            #   从全部WSI文件中筛选对应的坐标列表,返回相应的训练图像和label
            slideIDX, coord, target,h_value = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.patch_size[slideIDX],\
                                                    self.patch_size[slideIDX])).convert('RGB')
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
            if img.size != (224,224):
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    parser.add_argument('--train_lib', type=str, default='output/lib/512/cnn_train_data_lib.db', help='path to train MIL library binary')
    parser.add_argument('--val_lib', type=str, default='output/lib/512/cnn_val_data_lib.db', help='path to validation MIL library binary. If present.')
    parser.add_argument('--output', type=str, default='output/', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=12, help='number of epochs')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    # 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
    #parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.75, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=10, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    #parser.add_argument('--tqdm_visible',default = True, type=bool,help='keep the processing of tqdm visible or not, default: True')

    main(parser)
