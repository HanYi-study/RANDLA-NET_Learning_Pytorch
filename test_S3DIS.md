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
  ``` bash
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
``` python
from helper_tool import ConfigS3DIS as cfg
# 导入 S3DIS 数据集的配置信息（如类别数、采样参数、半径等）。这是 RandLA-Net 项目中的自定义配置模块。
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
# Network：主网络结构类（RandLA-Net）
# compute_loss：计算损失函数
# compute_acc：计算准确率
# IoUCalculator：用于计算每类及平均 IoU（交并比）
from s3dis_dataset import S3DIS, S3DISSampler
# S3DIS：数据集加载类，继承 torch.utils.data.Dataset
# S3DISSampler：测试时用的采样器，用于控制如何按块/点提取数据
import numpy as np
# 用于数值计算、矩阵操作（如点云数据、坐标、标签处理等）
import os, argparse
# os：操作系统接口，用于路径管理、创建文件夹等
# argparse：命令行参数解析模块
# from os.path import dirname, abspath, join  # 用于拼接路径

import torch
# PyTorch 主模块，用于张量操作、模型加载、推理等
from torch.utils.data import DataLoader
# 用于批量加载数据，控制 batch size、多线程等
import torch.nn as nn
# 包含神经网络构建常用模块（如 Conv、Linear、BatchNorm 等）
import torch.optim as optim
# 优化器模块，例如 Adam, SGD，尽管测试阶段可能不使用
import torch.nn.functional as F
# 包含常用函数如 relu, softmax, cross_entropy 等（一般用于前向传播中）
from sklearn.metrics import confusion_matrix
# 计算混淆矩阵，用于评估分类模型的性能（准确率、IoU等）
from helper_tool import DataProcessing as DP
# 项目中自定义的数据处理模块（如点云归一化、邻域查询、KNN等函数集合）
from helper_ply import write_ply
# 用于将点云预测结果保存为 .ply 格式文件，便于可视化
from datetime import datetime
# datetime：获取当前时间，用于记录日志等
import time
# time：计时与时间戳生成（例如输出路径命名）
import warnings
# 控制或忽略 Python 警告信息（如浮点精度、弃用警告等）
```
---
### 二、ArgumentParser参数解析
```python
parser = argparse.ArgumentParser()
# 创建一个 ArgumentParser 对象，argparse 是 Python 的标准库，用于从命令行中解析参数
# 这是准备添加命令行参数的前提
parser.add_argument('--checkpoint_path', default='output/checkpoint_Area_5.tar', help='Model checkpoint path [default: None]')
# 添加一个可选参数 --checkpoint_path，用于指定模型的预训练权重文件路径（.tar）
# default='output/checkpoint_Area_5.tar'：若用户未指定该参数，则默认使output/checkpoint_Area_5.tar
# help='...'：提供帮助信息，运行 python test_S3DIS.py --help 时会显示
# ps:实际上训练模型权重文件存放路径为/home/hy/projects/RandLA-Net-Pytorch-New/train_output/2025-07-12_01-48-28/checkpoint.tar
parser.add_argument('--log_dir', default='test_output', help='Dump dir to save modelcheckpoint [default: log]')
# 添加 --log_dir 参数，表示保存测试日志和可视化结果的目录路径
# 默认值为 'test_output'，输出文件（如 .ply, .txt, .log）通常会保存在这里
# 可通过命令行指定一个新的输出文件夹
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 2], -1 for cpu')
# 添加 --gpu 参数，用于指定模型测试时使用的 GPU 编号
# type=int：确保输入的是整数
# default=0：默认使用编号为 0 的 GPU（如 cuda:0）
# -1 代表使用 CPU 进行推理，这通常用于调试
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test,option: 1-6 [default: 6]')
# 设置测试区域编号
# 在 S3DIS 数据集中共有 6 个 Area（1~6），通常做交叉验证时会用其中一个 Area 作为测试集，其余作为训练集
# 默认是 5，表示“Area 5”用于测试，其它 Area 用于训练
FLAGS = parser.parse_args()
# 解析上述定义的所有参数，并将其保存在 FLAGS 对象中
# 调用时，比如 FLAGS.gpu 就可以获取命令行中指定的 GPU 编号，FLAGS.test_area 获取指定测试区域等
```
示例输入：  
```bash
python test_S3DIS.py \
    --checkpoint_path output/checkpoint_Area_3.tar \
    --log_dir results_area3_test \
    --gpu 1 \
    --test_area 3
```
- 加载 output/checkpoint_Area_3.tar 作为模型  
- 使用 GPU 1
- 指定 Area 3 为测试集
- 所有输出（日志、预测点云等）保存在 results_area3_test 文件夹中
---
### 三、模型测试准备与环境配置模块
该模块的作用：
1. 日志目录创建：创建一个用于保存测试日志和预测结果的文件夹
2. 日志写入工具函数定义：定义一个通用的 log_string() 函数，用于将信息写入日志文件并打印到终端
3. 测试数据加载：基于给定测试区域（Area），构建测试集并加载成 PyTorch 的 DataLoader
4. GPU/CPU 设备选择：根据参数 --gpu 和系统实际 GPU 情况，自动设置计算设备
5. 模型加载与恢复：从指定的 checkpoint 恢复模型和优化器参数，以用于后续的推理测试

输入说明：
- 从 argparse 中获取的参数：
    - log_dir：日志保存路径
    - test_area：测试区域（Area 1–6）
    - gpu：选择哪个 GPU，如果为 -1 则使用 CPU
- 模型 checkpoint 的路径 checkpoint_path（在此为写死路径，也可能是命令行传入）

输出说明：
- 创建测试日志文件（例如：test_output/2025-08-06_08-30-15/log_test_Area_5.txt）
- 创建预测输出保存目录：val_preds/
- 准备好运行环境 device

```python
# 1.创建日志目录与日志文件
LOG_DIR = FLAGS.log_dir
# 获取从命令行传入的日志根目录（如 'test_output'）
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
# 返回的是英国时间
# 拼接日志目录 + 当前时间（GMT时间），作为唯一的日志输出文件夹名
# 测试结束后生成了/home/hy/projects/RandLA-Net-Pytorch-New/test_output/2025-08-03_08-57-08/log_test_Area_5.txt
if not os.path.exists(LOG_DIR):
# 如果这个目录不存在，进入创建逻辑       
    os.makedirs(os.path.join(LOG_DIR, 'val_preds'))
    # 创建多级目录
    # 创建日志目录及其子目录 val_preds，用于保存验证集预测结果（如 .ply 文件等）
log_file_name = f'log_test_Area_{FLAGS.test_area:d}.txt'
# 构造日志文件名，log_test_Area_5.txt
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')
# 打开日志文件，使用追加写入模式 'a'，确保多轮日志可以连续写入

# 2. 定义日志输出函数
def log_string(out_str):
# 定义 log_string 函数，将日志同时写入文件和打印到终端
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # 保证每次都实时写入，不等缓冲区满
    print(out_str)

# 3. 加载测试集
dataset = S3DIS(FLAGS.test_area)
# 使用测试区域编号（如 Area 5）实例化原始点云数据集对象
test_dataset = S3DISSampler(dataset, 'validation')
# 将 dataset 包装成一个采样器（S3DISSampler），指定采样模式为 'validation'，用于验证阶段采样
test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=True,collate_fn=test_dataset.collate_fn)
# 构造 DataLoader，用于分 batch 遍历测试集
# shuffle=True 让每次遍历顺序打乱
# collate_fn 是该数据集特定的数据拼接方式（比如点云补零对齐等）

# 4. 配置设备（CPU / GPU）
if FLAGS.gpu >= 0:  # 如果使用GPU
# 判断是否使用 GPU（若为 -1 表示用户希望使用 CPU）
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    # 如果有可用 GPU，则使用指定编号（如 0、1）的 GPU 设备
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
    # 如果系统没安装 CUDA，强制使用 CPU 并给出警告
else:  # 如果不使用GPU
    FLAGS.gpu = torch.device('cpu')
    # 如果用户传入 --gpu=-1，也直接用 CPU
device = FLAGS.gpu
# 将最终选择的设备保存为变量 device，用于模型加载、数据转移等

# 5. 加载模型与优化器
net = Network(cfg)
# 初始化 RandLA-Net 网络结构，使用全局配置 cfg
net.to(device)
# 将模型移动到指定设备上（CPU 或 GPU）
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
# 定义 Adam 优化器，学习率来自配置 cfg.learning_rate

# 6. 加载训练好的模型权重
checkpoint_path = '/home/hy/projects/RandLA-Net-Pytorch-New/train_output/2025-07-12_01-48-28/checkpoint.tar'
# 指定模型 checkpoint 路径（这里是硬编码，通常应从 argparse 获取）
print(os.path.isfile(checkpoint_path))
# 打印该路径是否存在，作为调试提示
if checkpoint_path is not None and os.path.isfile(checkpoint_path):
# 如果路径存在，且确实是个文件，执行恢复模型操作
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # 加载 checkpoint 到 CPU 中，防止 GPU 加载失败（可 later 再转到 GPU）
    net.load_state_dict(checkpoint['model_state_dict'])
    # 恢复模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 恢复优化器参数（如果后续要继续 fine-tune）
    print("Model restored from %s" % checkpoint_path)
    # 打印提示模型加载成功
else:
   raise ValueError('CheckPointPathError')
# 如果路径不存在或文件缺失，直接报错终止程序
```
---
### 三、ModelTester 测试模块（Voting 推理 + 精度评估）
作用：
1. 用于在验证集上执行 多轮投票推理 (multi-vote inference)，提升预测的稳定性
2. 通过累积多个投票结果，对每个点的最终分类更平滑
3. 每次推理后计算准确率、混淆矩阵、IoU（Intersection over Union）等关键指标
4. 将最终的预测结果与 ground truth 一起保存为 .ply 文件供可视化
5. 最终目标是获得每个验证场景的分割效果评估

输入：
- dataset（S3DIS）：数据集对象，包含输入点云、标签、采样信息等
- num_vote（int）：投票次数，默认值为 100

输出：
- 控制台输出：
   - 每 step 的准确率 acc
   - 每轮投票后当前 mIoU 和各类 IoU
- 文件输出：
   - 每个场景的预测结果（.ply 文件）保存在 val_preds 目录中
   - 日志记录保存在 log_test_Area_*.txt

```python

class ModelTester:
# 初始化
    def __init__(self, dataset):
        self.test_probs = [np.zeros(shape=[l.shape[0], dataset.num_classes], dtype=np.float32)
       # 创建一个与验证集点数量和类别数一致的概率矩阵，用于累积投票结果。
       # self.test_probs[i][j] 表示第 i 个场景第 j 个点的预测类别概率
                           for l in dataset.input_labels['validation']]

#主测试函数
    def test(self, dataset, num_vote=100):
    # 主要目的是在测试集上评估模型性能，并通过投票策略（Voting）提高预测的稳定性和准确率
    # 投票机制：每个测试样本会被预测多次（num_vote=100），每次会有微小扰动或随机性，最终将多个预测结果融合（平滑）以获得更准确的最终结果
    # self: 表示类的实例
    # dataset: 测试使用的数据集，通常为 S3DISSampler 实例
    # num_vote: 投票次数，默认是 100 / 预测的次数
        # 初始化变量
        test_smooth = 0.95
        # test_smooth 是投票的平滑参数，用于控制每轮投票的加权融合程度
        # 越接近 1，说明之前的投票权重越大，越稳定，但响应慢
        # 越接近 0，说明当前预测的权重更大，变化更快但更敏感
        val_proportions = np.zeros(dataset.num_classes, dtype=np.float32)
        # val_proportions 用于统计验证集每个类别点的数量，其长度等于类别数
        # 计算每类点的数量（用于后续IoU加权）
        # 类型设为 float32 是为了后续计算 IoU 和权重时更方便
        i = 0
        # 初始化一个索引变量 i，用于标记非忽略类别的索引位置
        for label_val in dataset.label_values:
        # 遍历所有标签值，包括可用标签和被忽略的标签
        # dataset.label_values 是类别标签的完整列表（例如 [0, 1, 2, 3, ..., 12]）
            if label_val not in dataset.ignored_labels:
            # 只处理那些没有被忽略的标签，比如在 S3DIS 数据集中，ignored_labels 可能包含 0（表示背景或未标注点）
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                # 统计验证集中某一类别的点的总数量
                # dataset.val_labels 是验证集所有样本的标签集合（按文件划分）
                # labels == label_val 会得到一个布尔数组，表示每个点是否属于该类别
                # np.sum(labels == label_val) 得到该文件中属于 label_val 的点数
                # 再对所有文件求和，得到该类别在整个验证集中的点数
                # 最终结果存入 val_proportions[i] 中
                i += 1
  

        step_id = 0  # 用于计数测试过程中已经执行了多少个 batch
        epoch_id = 0  # 记录测试过程中进行的是第几轮投票（投票策略中每一轮都称为一次“epoch”），num_vote=100 表示将对每个点进行 100 次预测投票；epoch_id 会从 0 计数到 99
        last_min = -0.5  # 用于追踪“测试集采样器”当前所采样点云中 min_pos 的位置是否有变化，若无变化可能跳出循环

        while last_min < num_vote:
            stat_dict = {}
            net.eval() # set model to eval mode (for bn and dp)
            iou_calc = IoUCalculator(cfg)    

            for batch_idx, batch_data in enumerate(test_dataloader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                # cloud_idx = batch_data['cloud_inds']
                # point_idx = batch_data['input_inds']


                # Forward pass
                with torch.no_grad():
                    end_points = net(batch_data)

                loss, end_points = compute_loss(end_points, cfg, device)

                # stacked_probs = end_points['valid_logits'].cpu().numpy()          # logit值，还未经过归一化
                # stacked_labels = end_points['valid_labels'].cpu().numpy()
                # point_idx = end_points['input_inds'].cpu().numpy()
                # cloud_idx = end_points['cloud_inds'].cpu().numpy()

                # correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)        # 计算准确预测的点数
                # acc = correct / float(np.prod(np.shape(stacked_labels)))                    # 计算正确率
                # print('step' + str(step_id) + ' acc:' + str(acc))
                # stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                #                             cfg.num_classes])

                stacked_probs = end_points['valid_logits']         # logit值，还未经过归一化
                stacked_labels = end_points['valid_labels']
                point_idx = end_points['input_inds'].cpu().numpy()
                cloud_idx = end_points['cloud_inds'].cpu().numpy()

                correct = torch.sum(torch.argmax(stacked_probs, axis=1) == stacked_labels)        # 计算准确预测的点数
                acc = (correct / float(np.prod(stacked_labels.shape))).cpu().numpy()             # 计算正确率
                print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = torch.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                            cfg.num_classes])
                stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
                stacked_labels = stacked_labels.cpu().numpy()

                for j in range(np.shape(stacked_probs)[0]):     # 逐个batch进行计算
                    probs = stacked_probs[j, :, :]      # 当前batch的预测结果（分数）
                    p_idx = point_idx[j, :]             # 当前batch的点的序号
                    c_i = cloud_idx[j][0]               # 当前batch归属于哪个场景
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs # voting操作，平均多次推理的结果，让结果更稳定平滑
                step_id += 1

            # new_min = np.min(dataset.min_possibility['validation'])       注释掉是因为希望只推理一遍就行了
            new_min = 7.7
            log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                log_string('\nConfusion on sub clouds')                                 # 下面的结果是在网格采样后的子云的结果
                confusion_list = []

                num_val = len(dataset.input_labels['validation'])           # 验证集有多少个场景      

                for i_test in range(num_val):
                    probs = self.test_probs[i_test]                         # 取出第i_test个场景的vote后的结果
                    preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32) # 学习这种索引方式，索引中的数组的长度不一定要比被索引的小
                    labels = dataset.input_labels['validation'][i_test]     # 拿到第i_test个场景对应label                      

                    # Confs
                    confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # 计算该场景下的混淆矩阵（13*13）并追加保存为列表

                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)                 # 堆叠以后按列加起来 表示整个Area的混淆矩阵

                # Rescale with the right number of point per class      # 这里应该是根据正确点重新缩放混淆矩阵？
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)            # 混淆矩阵按行加起来就是每个类别分到的点数

                # Compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string(s + '\n')


                if int(np.ceil(new_min)) % 1 == 0:

                    # Project predictions
                    log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    proj_probs_list = []

                    for i_val in range(num_val):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.val_proj[i_val]                  # 取出第i_val个场景的原始点编号
                        probs = self.test_probs[i_val][proj_idx, :]         # 这里的编号很意思，跟之前生成这个val_proj有关。这一步其实就已经完成了采样后点云到原始点云之间的投影（结果预测）
                        proj_probs_list += [probs]                          # 将原始点云的预测结果保存在这个list中

                    # Show vote results
                    log_string('Confusion on full clouds')                  # 下面的结果是在网格采样前的结果
                    confusion_list = []
                    for i_test in range(num_val):
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)   # 根据结果（logit）求出最终的分类

                        # Confusion
                        labels = dataset.val_labels[i_test]     # 取出该场景的label
                        acc = np.sum(preds == labels) / len(labels) # 计算准确率
                        log_string(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc))

                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # 计算混淆矩阵
                        name = dataset.input_names['validation'][i_test] + '.ply'
                        write_ply(os.path.join(LOG_DIR, 'val_preds', name), [preds, labels], ['pred', 'label'])

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0)

                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_string('-' * len(s))
                    log_string(s)
                    log_string('-' * len(s) + '\n')
                    print('finished \n')
                    return                          # 到这里就结束了，只运行了一次（可能是效果比较好，只运行了一次，也就是测试的时候只有一个epoch）

            epoch_id += 1
            step_id = 0
            continue
        
        return



```
