# main_SemanticKITTI.py 与 main_S3DIS.py 的区别

这两个脚本都是 RandLA-Net 的训练主程序，但分别针对不同的数据集（SemanticKITTI 和 S3DIS），在数据加载、配置、日志、训练流程等方面有如下区别：

---

## 1. 针对的数据集不同

- **main_SemanticKITTI.py**：用于训练和验证 SemanticKITTI 数据集（自动驾驶场景，序列点云）。
- **main_S3DIS.py**：用于训练和验证 S3DIS 数据集（室内点云分割，按房间组织）。

---

## 2. 配置文件和数据集类不同

- **main_SemanticKITTI.py**
  - 配置：`from helper_tool import ConfigSemanticKITTI as cfg`
  - 数据集：`from semantic_kitti_dataset import SemanticKITTI`
- **main_S3DIS.py**
  - 配置：`from helper_tool import ConfigS3DIS as cfg`
  - 数据集：`from s3dis_dataset import S3DIS, S3DISSampler`

---

## 3. 数据加载方式不同

- **main_SemanticKITTI.py**
  - 直接用 `SemanticKITTI('training')` 和 `SemanticKITTI('validation')` 创建数据集对象。
  - DataLoader 直接用数据集对象，`collate_fn` 由数据集类提供。
  - 支持多线程加载（`num_workers=20`）。
- **main_S3DIS.py**
  - 先用 `S3DIS(FLAGS.test_area)` 创建基础数据集，再用 `S3DISSampler` 包装成训练/验证集。
  - DataLoader 用不同 batch_size，`collate_fn` 由采样器提供。
  - 没有显式指定 `num_workers`，使用默认。

---

## 4. 日志文件命名不同

- **main_SemanticKITTI.py**
  - 日志文件名固定为 `log_train_kitti.txt`。
- **main_S3DIS.py**
  - 日志文件名包含测试区域编号，如 `log_train_Area_5.txt`。

---

## 5. 训练参数和命令行参数不同

- **main_SemanticKITTI.py**
  - `--checkpoint_path` 默认 `output/checkpoint.tar`
  - `--max_epoch` 默认 100
  - `--gpu` 默认 0
- **main_S3DIS.py**
  - `--checkpoint_path` 默认 `None`
  - `--max_epoch` 默认 100
  - `--gpu` 默认 0
  - 多了 `--test_area` 参数，指定哪个 Area 用于测试

---

## 6. 训练流程和保存模型方式基本一致

- 两者都支持断点续训（加载 checkpoint）。
- 都在每个 epoch 后评估并保存当前最优模型到 `checkpoint.tar`。
- 都有详细的日志输出和 IoU 统计。

---

## 7. 其它细节

- **main_SemanticKITTI.py**：
  - DataLoader 支持多线程，适合大规模序列数据。
  - 日志目录和文件名更通用。
- **main_S3DIS.py**：
  - 禁用了 cudnn 加速（`torch.backends.cudnn.enabled = False`），适应大点云数据。
  - 日志文件包含测试区域信息，便于多区域实验管理。

---

## 总结表

| 脚本名                | 针对数据集      | 配置类                | 数据集类/采样器           | 日志文件名                | 特殊参数         | DataLoader线程 | 其它区别                |
|-----------------------|----------------|-----------------------|---------------------------|---------------------------|------------------|----------------|-------------------------|
| main_SemanticKITTI.py | SemanticKITTI  | ConfigSemanticKITTI   | SemanticKITTI             | log_train_kitti.txt       | 无               | 20             | 日志更通用              |
| main_S3DIS.py         | S3DIS          | ConfigS3DIS           | S3DIS + S3DISSampler      | log_train_Area_{n}.txt    | --test_area      | 默认           | 禁用cudnn，日志含区域信息 |

---

## 一句话总结

**main_SemanticKITTI.py 和 main_S3DIS.py 的核心训练流程类似，但分别针对不同点云分割数据集，数据加载、配置、日志和部分参数做了适配
