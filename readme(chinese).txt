## 基于耳道心音的低代价身份识别和认证系统

智能终端和线上支付的普及使数据安全和隐私保护面临了新的挑战。得益于生物特征的个体特异性，生物识别技术具有天然的安全性。生理信号由器官活动产生，它取决于遗传和基因，具备极强的个体特异性，因此基于生理信号的身份识别与认证具有独特优势。然而，生理信号的获取通常依赖专用设备，因而制约了此类技术在移动设备的应用。对此，本文通过可穿戴耳机获取耳道内由心脏跳动产生的耳道心音信号(Ear Canal Phonocardiogram, EarPCG)，提出基于耳道心音信号的低代价的身份识别和认证系统HeartID。

**matlab代码中包括**

1. 提出集合经验模态分解(EEMD)与小波阈值降噪(WTD)相结合的EarPCG信号消噪方法。该方法将非平稳的EarPCG信号分解为不同尺度的本征模态函数(IMF)，从多个尺度去除EarPCG中的噪声和干扰

2. 提出基于心跳周期分割的心率变异性消除算法，将EarPCG信号按心跳周期分割为样本从而去除人体心率变化的影响

3. 提取耳道心音的连续小波变换(CWT)特征

**python代码中包括**

1. 特征映射网络HeartNet，将低维的CWT特征映射到高维特征空间，从而提供了较大的个体区分。

2. 基于EarPCG信号的身份识别网络

3. 基于数据增强策略的身份认证模型

![image-20230607204556122](images/image-20230607204556122.png)

#### 项目代码使用说明

##### 1、数据采集

**设备**

![image-20230607204058041](images/image-20230607204058041.png)

```
run ./matlab/SoundRecorderDemo2.m
点击start开始采集，30s后自动结束
点击save保存数据
```

<img src="images/image-20230607204647792.png" alt="image-20230607204647792" style="zoom:50%;" />

**保存数据路径**

```
根目录
  └─原始数据
    ├─USERSxxx_01.mat
    ├─		...
    └─USERSxxx_xx.mat
 
K:.
├─Dataset
│  └─USERS015
│      ├─testset
│      ├─trainset
│      └─validset
├─eemd
│  └─USERS015
├─原始数据
└─同步数据
    └─USERS015
```



##### 2、数据处理和特征提取

```
点击开始处理
在弹窗中选择根目录（注意是根目录，不是数据的路径）
程序将在根目录下生成Dataset文件夹保存生成的cwt特征图片
图窗中将显示生成的cwt特征
```

**生成特征的路径**

```
根目录
  ├─Dataset
  │  └─USERSxxx
  │		...
  │  └─USERSxxx
  │      ├─testset
  │      ├─trainset
  │      └─validset
  ├─同步数据
  └─原始数据
```

##### 3、识别和认证

**身份识别**

```
将训练好的模型放入model文件夹
点击选择样本
在弹窗中选择想要测试的cwt样本
点击开始识别
```

**身份认证**

```
将训练好的模型放入models文件夹
点击选择样本
在弹窗中选择想要测试的cwt样本
点击开始认证
```

##### 4、模型训练

**身份识别**

```
在code\python\Identification文件夹下
运行 python train.py --dataset_path <数据集路径>
```

**身份认证**

```
在code\python\Authentication文件夹下
运行 python train.py --trainpath <训练集路径> --evalpath <验证集路径> --model_dir <模型保存路径>
```

##### 5、实验评估

**身份识别**

```
在code\python\Identification文件夹下
运行 python test.py --dataset_path <数据集路径>
```

**身份认证**

```
在code\python\Authentication文件夹下
运行 python eval.py --trainpath <训练集路径(为了选择阈值)> --evalpath <验证集路径> --model_dir <模型保存路径>
```

