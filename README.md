> **Clinical-grade computational pathology using weakl论文中对WSI图像MIL训练过程的实现**

- **论文地址**：https://www.nature.com/articles/s41591-019-0508-1
- **官方 Github**：https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019
- **参考 Github**：https://github.com/TankZhouFirst/clinical-grade-computational-pathology-using-weakly-supervised-deep-learning-on-whole-slide-images
****

## 论文代码的适用场景 ##

原论文是针对弱监督情况下对病理图像进行是否存在肿瘤区域(或其他异常区域，本例以肿瘤区域作为说明)的二分类判断,所谓弱监督是因为只有整张病理图片的标签但没有具体肿瘤区域的标注。在这个场景下，标记为肿瘤的病理图片至少会有一处区域是属于肿瘤的，而正常的病理图片则肯定是没有一处是肿瘤的。换言之，假设正常是0类，肿瘤是1类，其实在这两类在数值上是存在向下兼容的(1预测为0相对来说可以容忍，0预测为1则完全不能容忍。)，可以适当修改loss函数进行相应的加权或者惩罚(论文和官方代码没有提及这点)。而在官方代码中，指定使用预测为1类的概率(他们默认1类为肿瘤样本)作为top k比较的依据是可行的，因为一来这是两分类，二来这种两分类存在上下兼容关系，不管病理图片的实际标签是哪个分类，由于只对1类感兴趣，因此在整个MIL训练过程只选出预测1分类概率最高的top k作为整体代表是可行的。

然而如果是另一个场景，如果不是进行正常/肿瘤的判断，而是进行MSI/MSS这种不存在上下兼容关系的二分类或者多分类场景，即WSI的实际分类可能是基于图片中预测为所属类的区域最多者作为最终的预测结果的时候，上述默认只关注某一类预测概率的选择逻辑未必能适用。由于目前MSI和MSS这种基于基因表达上的差异能否通过WSI图像+深度学习的方法判断出来尚未得到确切论证(虽然已经有论文用MIL的思路尝试进行训练了)。在这种情况下，就不能直接用官方原来的代码思路进行训练了，我给出暂时的修改方案(见[General_Identify_MIL_train.py](#General_Identify_MIL_train.py))。

****

##  Installation ##

```pip install -r requirements.txt```
****


## 本人改进部分 ##

基于上述[TankZhouFirst](https://github.com/TankZhouFirst/clinical-grade-computational-pathology-using-weakly-supervised-deep-learning-on-whole-slide-images)和[官方](https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019)的代码,对**dataPrepare_for_CNN.py**和**MIL_train.py**的内部逻辑进行了比较大的调整,但是整体执行方式维持不变,以下是具体修改说明:


#### dataPrepare_for_CNN.py ####


1. 增加针对WSI图像**mpp**(0级下每像素代表多少微米,一般20X是0.5左右,40X是0.25左右)不同而自适应调整**patch size**的逻辑，并且**patch size**从保存固定值改为以列表形式保存，相关的读取修改已经在此前的提交中同步到`MIL_train_tqdm`(现已命名为`TUM_Identify_MIL_train`)中。

2. 原来的官方代码只是以设定的**patch size**对WSI文件进行全图坐标遍历，却没有考虑到实际上WSI中包含不少背景区域，全图无差别遍历会收集不少背景区域的坐标，而这些区域往往是无意义的。此次修改先对每一个WSI图片生成背景/组织区域的mask，然后只限定在组织区域内(基于mask)进行遍历，以生成需要保存的坐标组。

3. 考虑到即使只在组织区域进行遍历，采样出来的样本还是可能过大，导致筛选特征时耗时过长，不易于训练，因此我增设一个基于概率进行采样的机制。以计算每个WSI图片背景占全图的比例为采样概率，在遍历坐标并进行记录前先执行是否通过概率的判断，通过才保存该组坐标。如果是组织越多的图片，遍历时每一组坐标被采样的概率就越低，反之就越高，这样既能控制数量，也能实现均衡化采样。

4. 只保留采样的坐标组大于指定数量(目前是只保留>5，其实也可以设置只要求>0的)的WSI文件样本，原则上要求存到lib中的都能满足基本**top k**需求(我一般设置k为5或10)，但我已经修改了官方代码在训练时读取数据的逻辑，即使某张WSI采样下来的坐标组不足也可以对已有的坐标组进行重复采样保证满足设定的k的需求。

5. 运行该程序假设是有一个csv文件或者excel表格(这里暂命名为target文件，如果是excel表格的话需要自己改一下target文件的读取方法)。该文件格式如下：


| file_name        | class |
| :--------------: | :---: |
| ---------*.svs   |   0   |
| ---------*.svs   |   1   |
| .................|  ...  |


#### TUM_Identify_MIL_train.py ####

官方代码中，训练脚本原名为`MIL_train.py`,我这里将其重命名为`TUM_Identify_MIL_train.py`，以示专用于上述说明中提到的分类类别存在上下兼容关系的分类场景(比如判别正常/肿瘤)。

1. 首先在[TankZhouFirst](https://github.com/TankZhouFirst/clinical-grade-computational-pathology-using-weakly-supervised-deep-learning-on-whole-slide-images)注释的基础上对`group_argtopk`、`group_max`方法以及类`MILdataset`中的内部变量进行较为详尽的补充说明。

2. 修改`calc_accuracy`方法,取消原来的**err**和**fpr**，改为引入`sklearn.metrics`包的`balanced_acc`和`recall`作为新的统计指标。

3. 修改`inference`方法,增加**avg_prob**以及**whole_probably**的计算,前者在`tqdm`中显示.后者在**inferencing**后进行打印显示到屏幕。

4. 在`main`函数中,增加每一轮**top k**的保存,并且根据当前值和上一轮的值的变化而制定**early stopping**策略;同时修改每一轮训练后记录在日志上的变量;最后修改保存最佳模型的判断条件。

5. 针对`MILdataset`类,新增**patch size**的读取(每一个**slide**文件对应一个**patch size**数值);在`maketraindata`方法中新增复制采样的方法,为后面在`__getitem__`中采用不同参数的`adjust_hue`方法打下基础。颜色转换的操作放在在`__getitem__`中进行而不是在`transforms.Compose`中进行，确保在预测/计算概率的时候不进行任何变换，只在**train**的时候才引入变换。

6. 将原来日志输出通过`print`打印到屏幕改为使用`tqdm`库动态在屏幕显示输出，能实时监控每个batch对应评估指标的数值。效果大致如下：
![](https://github.com/BohriumKwong/using-weakly-supervised-deep-learning-on-whole-slide-images/blob/master/doc/images/created_gif.gif)



#### General_Identify_MIL_train.py ####

在[论文代码的适用场景](#论文代码的适用场景)中我提及到的第二个情况，就是当标签类别不存在上下兼容关系时，如果固定只选择某一类预测的概率进行top k去作为该WSI文件的代表的话，可能过于武断。因此我对论文官方代码的`inference`和`group_max`进行修改，以下是相关commit的说明:

1. 为`MILdatase`t新增label_mark的变量定义，该变量有助于在`inference`方法中及时判断当前读取器读取的图片对应的label实际分类。

2. 基于全新的思路调整`inference`方法，在**train**过程中，从不管实际分类提取1类该类作为**top k**排名改为根据图片所属lable进行概率的筛选,即标签为1类的图片选择1类的**top k**概率作为代表，0类的图片选择0类的**top k**概率作为代表；在**validation**过程中，就所有类别的概率都保留，进行下一步的分析。

3. 修改`group_max`方法，当前**slide文件**的预测结果改为以其所有region截图的预测概率的总和(或者平均值)取最大值者作为代表。

4. 为了减少程序中断而引起相关变量未能及时保留的惨剧，在每一轮训练过程调用`inference`后(不管是**train**还是**validation**)产生的**probs**和对应的**top k**都进行实时保存。

*但根据我的使用经验来看，以该方法作为模型的评估指标很容易出现过拟合的感觉，但换回官方原来的方法，指标的数据看上去会正常一些。请各位在使用时也要做好对比。*


#### MIL_train_v2.py ####

根据我们的实际使用经验，如果在MIL过程指定根据label来选取top k的话，很容易在开始的时候就让模型陷入过拟合中(其实也不难理解)，因此这种方法不适合从随机初始化参数未进行任何训练的模型，后来我们退回官方源代码的思想，只是`group_max`方法还是采用上述的[General_Identify_MIL_train.py](#General_Identify_MIL_train.py)定义的那样，暂定将此版本的脚本命名为**MIL_train_v2.py**。

#### MIL_train_v3.py ####

这个是在**MIL_train_v2.py**基础上修改的最终版本，也是我目前主要在用的版本。根据不同的实验比对发现，val数据集只有在**top 1**表现是最好的，而在train数据集中, **top k**中的**k**的数值越大，效果就越差,因此基本上是回到论文最初的思想。针对**MIL_train_v3.py**其他修改可以仔细查阅相关的commit log。

#### origin_lstm.py ####

这个是利用`pytorch`官方提供的`LSTM`类构建一个建议的基于LSTM的分类器(输入为图像经过网络全局pooling后输出的已flatten的特征)的代码。就目前来看，训练效果并不是很好。

#### LSTM_train.py ####
基于**MIL_train_v3.py**求出的数据集最佳的probs及其对应的IDX和gird信息(因为其中有可能会涉及到随机过采样,所以需要将当时生成的**dset.SlideIDX**和**dset.grid**也保存下来，详见脚本的注释和相关commit log)。通过修改dataloader的代码，实现新一轮的基于LSTM分类的训练流程，其中特征提取这一步放在自己写的dataloader中进行，不需要事前进行提取和相关存储。

****

> **论文中MIL过程原理图如下**

![](https://github.com/BohriumKwong/using-weakly-supervised-deep-learning-on-whole-slide-images/blob/master/doc/images/structure.png)

****

##  程序运行说明 ##
可以直接看官方的[README](https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019)文档，该文档也作为README文档放在./doc/下。


可以参考[TankZhouFirst](https://github.com/TankZhouFirst/clinical-grade-computational-pathology-using-weakly-supervised-deep-learning-on-whole-slide-images)用中文重新编写的文档，见[./doc/Clinical-grade computational pathology using weakly supervised deep learning on whole slide images.md](https://github.com/BohriumKwong/using-weakly-supervised-deep-learning-on-whole-slide-images/blob/master/doc/Clinical-grade%20computational%20pathology%20using%20weakly%20supervised%20deep%20learning%20on%20whole%20slide%20images.md)


> **Training**

在`dataPrepare_for_CNN.py`脚本中第14行设置好读入图片的文件路径(一级目录,默认后缀名是**svs**，但也有后缀名为**ndpi**的WSI文件)，以及在第138/145行设置好生成的db文件保存路径，然后直接运行`python dataPrepare_for_CNN.py`，则可完成对文件路径内所有WSI文件的用于MIL过程的坐标遍历结果(***_data_lib.db**文件)。

之后在`TUM_Identify_MIL_train.py`中main函数的传参列表中设置好相应参数，然后直接运行`python TUM_Identify_MIL_train.py`就可以开始训练，参数列表设置如下：

```python
parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='output/lib/512/cnn_train_data_lib.db', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='output/lib/512/cnn_val_data_lib.db', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='output/', help='name of output file')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 512)')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
# 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
parser.add_argument('--weights', default=0.79, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=5, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
```
其中训练执行顺序是先对全体train数据集进行infering，确定top k的index之后再对筛选后的数据进行训练，之后是对全体val数据集进行inferring，这样就完成一轮。执行时输出的结果如下：

Number of tiles: 46539
Epoch:1 train's inferencing: 100%|██████████| 1097/1097 [4:43:53<00:00, 43.75s/it, average mis probably - 0.1555]

Epoch:1 is trainng: 100%|██████████| 20/20 [08:39<00:00, 14.07s/it, acc - 0.6627, recall - 0.6279, fnr - 0.2857, loss - 0.6129]

Training        Epoch: [1/50] Acc: 0.617 Recall:0.659 Fnr:0.46 Loss: 0.656

Epoch:1 val's inferencing: 100%|██████████| 182/182 [2:17:09<00:00, 39.40s/it, average mis probably - 0.1712]

Validation  Epoch: [1/50]  acc - 0.5118, recall - 0.7013, fnr - 0.1639

Epoch:2 train's inferencing: 100%|██████████| 1097/1097 [4:56:36<00:00, 37.13s/it, average mis probably - 0.2188]

Epoch:2 is trainng: 100%|██████████| 20/20 [07:54<00:00, 12.94s/it, acc - 0.7475, recall - 0.6977, fnr - 0.1667, loss - 0.4709]

Training        Epoch: [2/50] Acc: 0.715 Recall:0.687 Fnr:0.236 Loss: 0.556

Epoch:2 val's inferencing: 100%|██████████| 182/182 [2:18:46<00:00, 44.21s/it, average mis probably - 0.1253]

Validation  Epoch: [2/50]  acc - 0.4372, recall - 0.5649, fnr - 0.3443

···


> **testing**
维持官方代码的运行方法，设置好相关配置后直接运行`python MIL_test.py`则可。

**输出如下**


![](https://github.com/BohriumKwong/using-weakly-supervised-deep-learning-on-whole-slide-images/blob/master/doc/images/result.png)
