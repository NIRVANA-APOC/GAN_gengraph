# 基于GAN的二次元头像生成模型

### 实验环境

```
python == 3.9.15
pytorch == 1.13.1
cuda == 11.7
CPU: 12th Gen Intel(R) Core(TM) i7-12700H
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

### 数据集下载

[飞桨-动漫头像数据集](https://aistudio.baidu.com/aistudio/datasetdetail/102512)

### 文件说明

```
config.py: 存放运行时的各项参数，可根据需要自行调整
model.py: 存放伸进网络结构的定义
train.py: 定义了神经网络的训练函数
generate.py: 定义了generate函数，可通过调用实现图像生成
main.py: 配置了简单的模型训练步骤
```

### 运行方式

进入主目录下运行一下命令即可

```
python main.py
```

若要调整迭代次数等参数在config.py文件中进行手动调整

### 实验结果

![](E:\homework\ArtificialIntelegent\exp\2\src\noise (2)\89.png)