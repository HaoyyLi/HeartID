## 基于耳道心音的低代价身份识别和认证系统

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

