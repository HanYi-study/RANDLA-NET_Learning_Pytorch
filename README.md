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
- 1）读取 S3DIS 原始数据集（`data/Stanford3dDataset_v1.2_Aligned_Version/`），每个房间一个文件夹，内含多个实例的 txt 文件（XYZRGB，没有标签，文件名或者某种方式可以区分不同实例）。
- 2）合并每个房间所有实例的 txt 文件，生成带标签的点云（XYZRGBL）：遍历每个房间文件夹，收集所有实例txt文件。读取每个实例txt文件，添加**标签列**，存储合并后的点云
- 3）对合并后的点云做**网格下采样**（如 0.04m）：
    - 首先将每个房间的所有txt读取合并为**N×7矩阵：xyz rgb label**并保存为ply文件；**原始数据合并后得到的是N×7维度(x,y,z,r,g,b,label)的.ply文件（保存在original_ply/*.ply），用于原始点云存档**
        - 为什么原始点云数据合并需要保留N×7维度的信息？
            - 首先是原始数据备份： original_ply/*.ply保留完整的原始信息（XYZRGB+label），用于调试或后续可能的重新处理。
            - 标签投影需要：proj.pkl中的标签来源于原始点云，需确保与下采样点的空间对应关系正确。
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
    - 下采样结果保存到input_0.040目录下。**网格下采样后得到的是N×6维度(x,y,z,r,g,b)的.ply文件（保存在input_0.040/*.ply），用于存储下采样点云**
        - 为什么下采样后ply是N×6？
            - 点云与标签分离：RandLA-Net将几何信息（xyz+rgb）和语义标签分开存储，input_0.040/* .ply仅保存点的几何和颜色特征（用于网络输入），input_0.040/*_proj.pkl保存原始标签及投影关系（用于训练时动态分配标签）
            - KDTree只需要坐标（xyz）
            - 标签一致性：标签通过proj.pkl中的proj_inds动态映射到下采样点，避免因下采样导致的标签歧义
- 4）建立 **KDTree** 并保存为 pkl：
    - 首先使用上一步生成的input_0.040/* .ply点云（N×6矩阵，包含xyzrgb）/ 经过了下采样的点云
    - 其次进行KDTree的构建：提取点云坐标xyz（忽略rgb），使用sklearn.neighbors.KDTree构建空间索引结构。
    - 然后是保存KDTree，使用pickle序列化KDTree对象，保存为*_KDTree.pkl文件。
    - 最后是关联投影索引，同时生成*_proj.pkl，记录原始点到下采样点的最近邻索引（利用KDTree加速查询）。
    - 经过以上处理后输出了```input_0.040/*_KDTree.pkl```（KDTree对象文件，支持高效KNN查询，包含了KDTree索引，用于加速邻域查询）和```input_0.040/*_proj.pkl```（投影索引文件，内容包含了投影索引+原始标签，用于表爱你动态分配）
- 5）保存原始点到下采样点的最近邻投影索引和标签为 pkl。
    - 首先输入数据：原始点云（original_ply/* .ply，N×7，包含 x y z r g b label）和下采样点云（input_0.040/*.ply，M×6，仅 x y z r g b）
    - 计算最近邻投影索引：使用 KDTree（基于下采样点云构建）查询每个原始点的最近邻下采样点索引
    - 保存标签和投影索引：原始标签直接取自original_ply的第七列，然后二者存储为.pkl文件。
- **投影索引**：
    - 什么是投影索引：投影索引（Projection Index）是一个一维数组，记录原始点云中每个点对应的下采样点云中的最近邻点的索引。
    - 形状：[N_original]（N_original = 原始点云点数）
    - 内容：每个元素是下采样点云中的点的索引（如proj_inds[0] = 3表示原始点0映射到下采样点3）。
    - 为什么需要投影索引：解决下采样导致的标签与点云不对齐问题。
        - 下采样会丢失点：网格下采样后，多个原始点可能合并为一个下采样点，导致原始标签无法直接使用。
        - 标签传递：通过投影索引，将原始标签正确分配给下采样点（如多数投票）。
        - 预测还原：测试时需将下采样点的预测结果插值回原始点云（依赖投影关系）。
    - 如何计算投影索引
        - 构建下采样点的KDTree
        - 查询原始点的最近邻下采样点
        - 保存为proj.pkl
        - ```text
          原始点云: A(0), B(1), C(2), D(3)      标签: [0, 1, 2, 3]
          下采样点云: X, Y, Z                     (A,B→X; C→Y; D→Z)
          投影索引: [0, 0, 1, 2]                (A→X, B→X, C→Y, D→Z)
          ```
    - 为什么两次计算最近邻投影索引？
        - 第一次投影索引（用于训练时的标签分配）：下采样后生成input_0.040/*_proj.pkl时进行计算 / 从原始点云，进行下采样生成下采样点时 / 输出内容：proj_inds, labels / 用途：训练时动态分配标签（如多数投票）
        - 第二次投影索引（用于测试时的预测还原）：测试时加载input_0.040/*_proj.pkl / 从下采样点反向推得原始点云 / 输出内容：下采样点预测结果插值到原始点，或者说将网络输出的下采样点预测结果扩展回原始分辨率 / 用途：测试阶段将预测结果还原到原始点云（可视化或计算mIoU）

### 1.3 输入数据
- `data/Stanford3dDataset_v1.2_Aligned_Version/Area_x/room_y/Annotations/*.txt`

### 1.4 输出数据
- `data/original_ply/*.ply`：合并后的原始点云（XYZRGBL）
- `data/input_0.040/*.ply`：下采样点云（XYZRGB）
- `data/input_0.040/*_KDTree.pkl`：KDTree 对象，包含了KDTree索引，支持KNN高效查询
- `data/input_0.040/*_proj.pkl`：原始点到下采样点的投影索引和标签

---

## 2. 数据集加载与采样

### 2.1 负责文件
- `s3dis_dataset.py`
- `helper_tool.py`（数据处理与采样工具）

### 2.2 主要操作
- **数据集加载**：通过 `S3DIS` 类加载预处理后的点云、标签、KDTree、投影索引等。
    - 首先读取文件：从存储的数据文件（如下采样后的.ply或.npy文件）中读取点云数据，包括点的坐标（xyz）、颜色（rgb）、标签（L）等
    - 然后统一数据结构：通常会将每个房间的点云数据加载到内存，构建为统一的numpy数组或pytorch tensor，便于后续批处理
    - 最后进行索引与映射：还会加载KDTree、点的索引映射等辅助数据，用于快速孔家搜索和数据增强
- **采样/空间均匀采样**：通过 `S3DISSampler` 实现空间均匀采样（spatially regular sampling），每次采样一批点用于训练/验证。
    - 采用概率机制，每次从点云中抽取一个中心点及其邻域，保证采样点云块在空间上的均匀分布，增强训练泛化。

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
