## **Research on low-cost identification and authentication based on Ear Canal Phonocardiogram**

![image-20230607204556122](images/image-20230607204556122.png)

#### Code Implementation Notes

##### 1、Data collection

**device**

![image-20230607204058041](images/image-20230607204058041.png)

```
run ./matlab/SoundRecorderDemo2.m
click start，automatically end after 30s
click save to save the data(name:USERS***_xx)
```

<img src="images/image-20230607204647792.png" alt="image-20230607204647792" style="zoom:50%;" />

**Dir Tree**

```
root
  └─原始数据
    ├─USERSxxx_01.mat
    ├─		...
    └─USERSxxx_xx.mat
```



##### 2、Data Processing and Feature Extraction

```
click 开始处理
choose root path（note: this is root path but not the data path）
The program will generate a Dataset folder in the root directory to save the generated cwt feature image
The generated cwt features will be displayed in the figure
```

**Path to CWT Feature**

```
root
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

##### 3、Identification and Authentication

**Identification**

```
Put the trained model into the model folder
Click to 选择样本
Select the cwt sample you want to test in the pop-up window
Click to 开始识别
```

**Authentication**

```
Put the trained model into the models folder
Click to 选择样本
Select the cwt sample you want to test in the pop-up window
Click to 开始认证
```

##### 4、Train model

**Identification**

```
In the code\python\Identification folder
run python train.py --dataset_path <dataset path>
```

**Authentication**

```
In the code\python\Authentication folder
run python train.py --trainpath <trainset> --evalpath <validset> --model_dir <model path>
```

##### 5、Test model

**Identification**

```
In the code\python\Identification folder
run python test.py --dataset_path <path of the dataset>
```

**Authentication**

```
In the code\python\Authentication folder
run python eval.py --trainpath <trainset(to choose threshold)> --evalpath <testset> --model_dir <model path>
```

