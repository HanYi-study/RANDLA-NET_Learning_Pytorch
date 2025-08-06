# test_S3DIS.py文件解析  

## 目前存在的问题 ；

---
## 主体功能概览：
test_S3DIS.py 是用于 评估 RandLA-Net 在 S3DIS 数据集上性能 的脚本，实现了 加载模型 ➝ 数据推理 ➝ 输出结果 ➝ 计算评估指标 的完整测试流程。  

1. 加载模型权重：加载指定路径下的预训练 .ckpt 或 .tar 权重文件。  
2. 设置测试区域：支持设置某一个 Area（1~6）作为测试区域，其他区域作为训练区域（标准 S3DIS cross-validation 策略）。
3. 构建数据加载器：使用 S3DIS 和 S3DISSampler 读取并组织测试数据。
4. 推理预测语义标签：遍历测试数据，调用模型进行前向推理，获取每个点的预测类别。
5. 可视化预测结果：将预测结果保存为 .ply 格式，可用于可视化工具查看。
6. 计算评估指标：统计混淆矩阵，最终输出每类 IoU、mean IoU（mIoU）和 overall accuracy（OA）。  
- 输入：
    - 模型权重文件的存放路径（checkpoint_path）：例如/home/hy/projects/RandLA-Net-Pytorch-New/train_output/2025-07-12_01-48-28/checkpoint.tar / 即为训练结束后生成的模型权重文件的路径
    - 测试的Area编号（test_area）：作为控制台输入的参数传入
    - S3DIS 格式的数据
- 输出：
    - .ply 文件：每个 block 的预测结果（点、RGB、预测标签）
    - log_test_Area_X.txt：测试日志，包括 IoU、OA 等，可以在log_dir中指定（默认生成的测试日志存放路径为test_output）
    - 控制台输出评估信息
- 在控制台中：
  ```
  python test_S3DIS.py \
  --checkpoint_path ./log_6fold/checkpoint.tar \
  --log_dir ./test_output \
  --test_area 5 \
  --gpu 0

  >可以实现：
   使用 GPU 0；
   对第 5 区进行测试；
  使用 ./log_6fold/checkpoint.tar 模型；
  把预测结果和日志输出到 ./test_output/。  

## 模块分解：
### 一、模块/包的导入
```
from helper_tool import ConfigS3DIS as cfg
# 
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from s3dis_dataset import S3DIS, S3DISSampler
import numpy as np
import os, argparse
# from os.path import dirname, abspath, join

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
from helper_ply import write_ply
from datetime import datetime
import time
import warnings
```
