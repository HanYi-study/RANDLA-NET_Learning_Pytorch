# RANDLA-NET_Learning_Pytorch
原github网址：https://github.com/liuxuexun/RandLA-Net-Pytorch-New/tree/master

# RandLA-NET-PYTORCH-NEW 项目目录结构（截至 2025-08-04）

```
RandLA-NET-PYTORCH-NEW/
├── __pycache__/                        # Python 编译缓存
│   ├── helper_ply.cpython-37.pyc
│   ├── helper_tool.cpython-37.pyc
│   ├── pytorch_utils.cpython-37.pyc
│   ├── RandLANet.cpython-37.pyc
│   └── s3dis_dataset.cpython-37.pyc
│
├── .idea/                              # VSCode/PyCharm 工程配置
│   ├── inspectionProfiles/
│   ├── misc.xml
│   ├── modules.xml
│   ├── RandLA-Net-Pytorch-New.iml
│   ├── vcs.xml
│   └── workspace.xml
│
├── data/                               # 数据目录
│   ├── input_0.040/
│   ├── original_ply/
│   └── Stanford3dDataset_v1.2_Aligned_Version/
│
├── previous_license/                   # 历史许可证（可忽略）
│
├── test_output/                        # 推理结果输出目录
│   ├── 2025-08-03_08-57-08/
│   │   └── val_preds/
│   │       └── log_test_Area_5.txt
│
├── train_output/
│   └── 2025-07-12_01-48-28/
│       ├── checkpoint.tar
│       └── log_train_Area_5.txt
│
├── utils/                              # 工具与数据处理模块
│   ├── cpp_wrappers/
│   ├── meta/
│   ├── nearest_neighbors/
│   │   ├── knn.cpp / knn.h / knn.o
│   │   ├── KDTreeTableAdaptor.h
│   │   ├── nanoflann.hpp
│   │   ├── knn.pyx
│   │   ├── knn_.cxx / knn_.h
│   │   ├── setup.py
│   │   ├── test.py
│   │   └── lib/python/
│   ├── 6_fold_cv.py
│   ├── data_prepare_s3dis.py
│   ├── data_prepare_semantic3d.py
│   ├── data_prepare_semantickitti.py
│   ├── download_semantic3d.sh
│   ├── semantic-kitti.yaml
│
├── compile_op.sh                       # 编译 CUDA 操作模块脚本（重复出现在根目录？）
├── cuda_11.6.0_510.39.01_linux.run     # CUDA 安装包（仅测试参考）
├── helper_ply.py                       # PLY 文件操作工具（可能重复）
├── helper_tool.py                      # 常用数据处理函数（可能重复）
├── job_for_testing.sh                  # Shell脚本：模型测试任务
├── LICENSE
├── main_S3DIS.py
├── main_SemanticKITTI.py
├── nvidia-compute-utils-550_xxx.deb
├── pytorch_utils.py
├── RandLANet.py
├── README.md
├── s3dis_dataset.py
├── semantic_kitti_dataset.py
├── test_S3DIS.py
└── test_SemanticKITTI.py
```

