# RANDLA-NET_Learning_Pytorch
原github网址：https://github.com/liuxuexun/RandLA-Net-Pytorch-New/tree/master

# RandLA-NET-PYTORCH-NEW 项目目录结构

```
RandLA-Net-Pytorch-New/
├── __pycache__/                        # Python 编译缓存文件夹
│   ├── helper_ply.cpython-37.pyc       # helper_ply.py 的编译缓存
│   ├── helper_tool.cpython-37.pyc      # helper_tool.py 的编译缓存
│   ├── pytorch_utils.cpython-37.pyc    # pytorch_utils.py 的编译缓存
│   ├── RandLANet.cpython-37.pyc        # RandLANet.py 的编译缓存
│   └── s3dis_dataset.cpython-37.pyc    # s3dis_dataset.py 的编译缓存
│
├── .idea/                              # PyCharm/IDEA 工程配置文件夹
│   ├── inspectionProfiles/             # 代码检查配置
│   ├── misc.xml                        # 工程配置
│   ├── modules.xml                     # 模块配置
│   ├── RandLA-Net-Pytorch-New.iml      # 工程模块文件
│   ├── vcs.xml                         # 版本控制配置
│   └── workspace.xml                   # 工作区配置
│
├── data/                               # 数据目录
│   ├── input_0.040/                    # 下采样后点云、KDTree、投影索引等
│   ├── original_ply/                   # 合并后的原始点云 ply 文件
│   └── Stanford3dDataset_v1.2_Aligned_Version/ # S3DIS 原始数据集
│
├── previous_license/                   # 历史许可证和说明文档
│   ├── LICENSE                         # 旧版许可证
│   └── README.md                       # 说明文档
│
├── test_output/                        # 推理/测试输出目录
│   ├── 2025-08-03_08-57-08/            # 按时间戳分文件夹保存本次测试结果
│   │   └── val_preds/                  # 验证集预测结果
│   │       └── log_test_Area_5.txt     # 测试日志
│
├── train_output/                       # 训练输出目录
│   └── 2025-07-12_01-48-28/            # 按时间戳分文件夹保存本次训练结果
│       ├── checkpoint.tar              # 训练断点模型权重
│       └── log_train_Area_5.txt        # 训练日志
│
├── utils/                              # 工具与数据处理模块
│   ├── cpp_wrappers/                   # C++/CUDA 加速模块源码
│   ├── meta/                           # 元数据（如类别名、路径等）
│   ├── nearest_neighbors/              # 最近邻查找相关 C++/Python 实现
│   │   ├── knn.cpp / knn.h / knn.o     # KNN 算法源码及编译文件
│   │   ├── KDTreeTableAdaptor.h        # KDTree 适配器头文件
│   │   ├── nanoflann.hpp               # nanoflann KDTree 库
│   │   ├── knn.pyx                     # Cython 封装
│   │   ├── knn_.cxx / knn_.h           # 生成的 C++ 文件
│   │   ├── setup.py                    # 编译脚本
│   │   ├── test.py                     # 测试脚本
│   │   └── lib/python/                 # 编译生成的 Python 库
│   ├── 6_fold_cv.py                    # S3DIS 六折交叉验证脚本
│   ├── data_prepare_s3dis.py           # S3DIS 数据预处理脚本
│   ├── data_prepare_semantic3d.py      # Semantic3D 数据预处理脚本
│   ├── data_prepare_semantickitti.py   # SemanticKITTI 数据预处理脚本
│   ├── download_semantic3d.sh          # 下载 Semantic3D 数据集脚本
│   ├── semantic-kitti.yaml             # SemanticKITTI 标签映射配置
│
├── compile_op.sh                       # CUDA/C++ 加速模块编译脚本
├── cuda_11.6.0_510.39.01_linux.run     # CUDA 安装包（仅供环境搭建参考）
├── helper_ply.py                       # PLY 点云文件读写工具
├── helper_tool.py                      # 常用数据处理与配置工具
├── job_for_testing.sh                  # Shell 脚本：批量模型测试任务
├── LICENSE                             # 项目许可证
├── main_S3DIS.py                       # S3DIS 数据集训练主程序
├── main_SemanticKITTI.py               # SemanticKITTI 数据集训练主程序
├── nvidia-compute-utils-550_xxx.deb    # NVIDIA 驱动相关安装包（仅供环境搭建参考）
├── pytorch_utils.py                    # PyTorch 相关辅助工具
├── RandLANet.py                        # RandLA-Net 网络结构与训练核心
├── README.md                           # 项目说明文档
├── s3dis_dataset.py                    # S3DIS 数据集 Dataset 类
├── semantic_kitti_dataset.py           # SemanticKITTI 数据集 Dataset 类
├── test_S3DIS.py                       # S3DIS 数据集测试/推理脚本
└── test_SemanticKITTI.py               # SemanticKITTI 数据集测试/推
```
08/04创建  
08/10更新  
不包含学习代码用的各.md文件

---

# RandLA-Net (S3DIS) 完整运行流程总结

本流程以 S3DIS 数据集为例，详细说明 RandLA-Net 从数据预处理到训练、测试的每一步，涉及的文件、输入输出数据及操作内容。

---

## 1. 数据预处理

### 1.1 负责文件
- `utils/data_prepare_s3dis.py`

### 1.2 主要操作
- 读取 S3DIS 原始数据集（`data/Stanford3dDataset_v1.2_Aligned_Version/`），每个房间一个文件夹，内含多个实例的 txt 文件（XYZRGB，没有标签，文件名或者某种方式可以区分不同实例）。
- 合并每个房间所有实例的 txt 文件，生成带标签的点云（XYZRGBL）：遍历每个房间文件夹，收集所有实例txt文件。读取每个实例txt文件，添加标签列，存储合并后的点云
- 对合并后的点云做**网格下采样**（如 0.04m）：
    - 首先将每个房间的所有txt读取合并为N×7矩阵（xyz rgb label）并保存为ply文件；
    - 调用DP.grid_sub_sampling()函数（DP为utils.tf_ops或utils.cpp_wrappers下的点云处理库）实现网格采样；
        - 空间网格划分：
            - 首先，将整个点云空间按照0.04米为边长划分为一个个立方体网格（体素，cube/grid）
            - 每个点 (x, y, z) 通过除以 0.04 并向下取整（floor），落入某一个网格单元。
        - 网格内点聚合：
            - 对每个网格，统计所有落入该网格的点。
            - 常见聚合方式：取网格内所有点的坐标均值（质心）作为代表点
            - 对应的颜色、标签也通常采用均值或多数投票
        - 输出下采样点云：
            - 每个有点的网格只输出一个代表点（大大减少点数，稀疏化点云）
    - 下采样结果保存到input_0.040目录下（ply格式，x y z red green blue class）
- 建立 KDTree 并保存为 pkl。
- 保存原始点到下采样点的最近邻投影索引和标签为 pkl。

### 1.3 输入数据
- `data/Stanford3dDataset_v1.2_Aligned_Version/Area_x/room_y/Annotations/*.txt`

### 1.4 输出数据
- `data/original_ply/*.ply`：合并后的原始点云（XYZRGBL）
- `data/input_0.040/*.ply`：下采样点云（XYZRGBL）
- `data/input_0.040/*_KDTree.pkl`：KDTree 对象
- `data/input_0.040/*_proj.pkl`：原始点到下采样点的投影索引和标签

---

## 2. 数据集加载与采样

### 2.1 负责文件
- `s3dis_dataset.py`
- `helper_tool.py`（数据处理与采样工具）

### 2.2 主要操作
- 通过 `S3DIS` 类加载预处理后的点云、标签、KDTree、投影索引等。
- 通过 `S3DISSampler` 实现空间均匀采样（spatially regular sampling），每次采样一批点用于训练/验证。

### 2.3 输入数据
- `data/input_0.040/*.ply`
- `data/input_0.040/*_KDTree.pkl`
- `data/input_0.040/*_proj.pkl`

### 2.4 输出数据
- 采样得到的点云 batch（点坐标、颜色、标签、索引等），供 DataLoader 使用。

---

## 3. 数据加载与批处理

### 3.1 负责文件
- `main_S3DIS.py`
- `s3dis_dataset.py`（`collate_fn`）

### 3.2 主要操作
- 使用 PyTorch `DataLoader` 加载采样后的 batch 数据。
- `collate_fn` 组装多层 KNN 索引、下采样索引、上采样索引等，形成网络输入格式。

### 3.3 输入数据
- 采样得到的点云 batch

### 3.4 输出数据
- 组装好的 batch 字典（包含 `xyz`, `features`, `labels`, `neigh_idx`, `sub_idx`, `interp_idx`, `input_inds`, `cloud_inds` 等）

---

## 4. 模型定义与训练

### 4.1 负责文件
- `RandLANet.py`（模型结构、损失、评估等）
- `main_S3DIS.py`（训练主程序）

### 4.2 主要操作
- 构建 RandLA-Net 网络结构。
- 前向传播，计算损失（`compute_loss`）、准确率（`compute_acc`）、IoU（`IoUCalculator`）。
- 反向传播与参数更新。
- 日志记录与模型保存。

### 4.3 输入数据
- DataLoader 输出的 batch 字典

### 4.4 输出数据
- 训练日志（如 `train_output/2025-07-12_01-48-28/log_train_Area_5.txt`）
- 断点模型权重（如 `train_output/2025-07-12_01-48-28/checkpoint.tar`）

---

## 5. 验证与测试

### 5.1 负责文件
- `test_S3DIS.py`
- `RandLANet.py`
- `s3dis_dataset.py`

### 5.2 主要操作
- 加载训练好的模型权重。
- 对验证/测试集进行推理，支持多次投票平滑预测。
- 将下采样点云的预测结果投影回原始点云。
- 计算混淆矩阵、IoU、准确率等指标。
- 保存预测结果（ply 文件）和测试日志。

### 5.3 输入数据
- 训练好的模型权重
- 预处理后的测试集数据（同第2步）

### 5.4 输出数据
- 测试日志（如 `test_output/2025-08-03_08-57-08/val_preds/log_test_Area_5.txt`）
- 预测结果 ply 文件（如 `test_output/2025-08-03_08-57-08/val_preds/*.ply`）

---

## 6. 可选：交叉验证与批量测试

### 6.1 负责文件
- `utils/6_fold_cv.py`（六折交叉验证）
- `job_for_testing.sh`（批量测试脚本）

### 6.2 主要操作
- 自动化多 Area 交叉验证训练与测试
- 批量提交测试任务

---

# 总结流程图

1. **数据预处理**  
   - `utils/data_prepare_s3dis.py`  
   - 输入：原始 txt  
   - 输出：ply、KDTree、proj.pkl

2. **数据集加载与采样**  
   - `s3dis_dataset.py`, `helper_tool.py`  
   - 输入：预处理数据  
   - 输出：采样 batch

3. **数据加载与批处理**  
   - `main_S3DIS.py`, `s3dis_dataset.py`  
   - 输入：采样 batch  
   - 输出：网络输入 batch

4. **模型训练**  
   - `RandLANet.py`, `main_S3DIS.py`  
   - 输入：网络输入 batch  
   - 输出：日志、模型权重

5. **验证与测试**  
   - `test_S3DIS.py`, `RandLANet.py`  
   - 输入：模型权重、测试数据  
   - 输出：预测结果、日志

6. **交叉验证/批量测试（可选）**  
   - `utils/6_fold_cv.py`, `job_for_testing.sh`

---

**每一步都严格依赖前一步的输出数据和相关脚本文件，确保数据流和功能链路完整 。**

更新至2025/08//10  **有待完善**
