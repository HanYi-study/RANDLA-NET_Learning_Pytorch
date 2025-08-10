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
