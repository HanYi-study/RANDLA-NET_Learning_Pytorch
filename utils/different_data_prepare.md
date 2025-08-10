# 三个点云数据预处理脚本的区别与对比

这三个文件都是点云分割任务的数据预处理脚本，但它们分别针对不同的数据集，处理流程和细节略有不同。下面是它们的主要区别和各自特点：

---

## 1. data_prepare_s3dis.py

**针对数据集**：S3DIS

- **输入数据**：Stanford Large-Scale 3D Indoor Spaces (S3DIS) 数据集，每个房间一个文件夹，内含多个 txt（每个实例一个 txt，内容为 XYZRGB）。
- **主要流程**：
  - 合并每个房间所有实例的 txt 文件，生成带标签的点云（XYZRGBL）。
  - 对合并后的点云做网格下采样（如 0.04m）。
  - 保存原始点云和下采样点云为 ply 文件。
  - 建立 KDTree 并保存为 pkl。
  - 保存原始点到下采样点的最近邻投影索引和标签为 pkl。
- **输出**：
  - `original_ply/xxx.ply`（原始点云）
  - `input_0.040/xxx.ply`（下采样点云）
  - `input_0.040/xxx_KDTree.pkl`
  - `input_0.040/xxx_proj.pkl`

---

## 2. data_prepare_semantickitti.py

**针对数据集**：SemanticKITTI

- **输入数据**：SemanticKITTI 数据集，按 sequence（00~21）组织，每个 sequence 下有 velodyne（点云）、labels（标签）。
- **主要流程**：
  - 读取每帧点云（bin），读取标签（label），用 yaml 文件做标签映射。
  - 对每帧点云做网格下采样（如 0.06m）。
  - 保存下采样点云和标签为 npy。
  - 建立 KDTree 并保存为 pkl。
  - 保存原始点到下采样点的最近邻投影索引为 pkl（验证集和测试集都保存）。
- **输出**：
  - `velodyne/*.npy`（下采样点云）
  - `labels/*.npy`（下采样标签）
  - `KDTree/*.pkl`
  - `proj/*.pkl`（投影索引）

---

## 3. data_prepare_semantic3d.py

**针对数据集**：Semantic3D

- **输入数据**：Semantic3D 数据集，每个场景一个 txt 文件（XYZRGBI），有的有 label 文件。
- **主要流程**：
  - 读取 txt 点云，若有 label 则读取标签。
  - 先做一次细粒度下采样（0.01m），再做一次粗粒度下采样（如 0.06m）。
  - 保存原始点云和下采样点云为 ply 文件。
  - 建立 KDTree 并保存为 pkl。
  - 保存原始点到下采样点的最近邻投影索引和标签为 pkl。
- **输出**：
  - `original_ply/xxx.ply`（原始点云）
  - `input_0.060/xxx.ply`（下采样点云）
  - `input_0.060/xxx_KDTree.pkl`
  - `input_0.060/xxx_proj.pkl`

---

## 总结对比

| 脚本文件                        | 针对数据集      | 输入格式/结构           | 主要处理流程                   | 输出内容（格式/结构）         |
|----------------------------------|----------------|------------------------|-------------------------------|------------------------------|
| data_prepare_s3dis.py            | S3DIS          | 房间/实例txt+RGB       | 合并、下采样、KDTree、投影     | ply, pkl, proj.pkl           |
| data_prepare_semantickitti.py    | SemanticKITTI  | sequence/velodyne+label| 下采样、KDTree、标签映射、投影 | npy, pkl, proj.pkl           |
| data_prepare_semantic3d.py       | Semantic3D     | 场景txt(+label)        | 下采样两次、KDTree、投影       | ply, pkl, proj.pkl           |

- **S3DIS/Semantic3D**：以房间/场景为单位，合并实例，ply 格式为主。
- **SemanticKITTI**：以序列和帧为单位，npy 格式为主，标签需 remap。
- **三者都做了下采样、KDTree、投影索引等，输出结构和文件夹命名略有不同。**

---

## 一句话总结

这三个脚本分别针对 S3DIS、SemanticKITTI、Semantic3D 三个主流点云分割数据集，核心流程类似（下采样、KDTree、投影），但输入输出格式和细节处理根据各自数据集结构做了适配。
