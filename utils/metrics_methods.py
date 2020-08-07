# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:40:37 2020

@author: Bohrium.Kwong
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

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