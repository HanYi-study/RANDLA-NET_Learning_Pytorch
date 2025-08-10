# test_S3DIS.py 与 test_SemanticKITTI.py 的区别

这两个脚本都是 RandLA-Net 的测试/推理脚本，但分别针对 S3DIS 和 SemanticKITTI 两个不同的数据集。它们在数据加载、模型加载、推理流程、输出处理等方面有如下区别：

---

## 1. 针对的数据集不同

- **test_S3DIS.py**：用于 S3DIS（室内点云分割）数据集的测试与评估。
- **test_SemanticKITTI.py**：用于 SemanticKITTI（自动驾驶点云分割）数据集的测试与评估。

---

## 2. 配置与数据集类不同

- **test_S3DIS.py**
  - 配置类：`ConfigS3DIS`
  - 数据集类：`S3DIS`, `S3DISSampler`
- **test_SemanticKITTI.py**
  - 配置类：`ConfigSemanticKITTI`
  - 数据集类：`SemanticKITTI`

---

## 3. 命令行参数不同

- **test_S3DIS.py**
  - `--checkpoint_path`：模型权重路径
  - `--log_dir`：日志输出目录
  - `--gpu`：GPU编号
  - `--test_area`：指定测试的 Area（1-6）
- **test_SemanticKITTI.py**
  - `--checkpoint_path`：模型权重路径
  - `--log_dir`：日志输出目录
  - `--gpu`：GPU编号
  - `--gen_pseudo`：是否生成伪标签
  - `--retrain`：是否用伪标签重新训练
  - `--test_area`：指定测试序列编号（如 '08'）

---

## 4. 数据加载方式不同

- **test_S3DIS.py**
  - 先用 `S3DIS` 加载数据，再用 `S3DISSampler` 采样，最后用 `DataLoader` 组织 batch。
- **test_SemanticKITTI.py**
  - 直接用 `SemanticKITTI` 加载测试集，`DataLoader` 组织 batch。

---

## 5. 模型加载与设备设置

- 两者都支持 GPU/CPU 自动切换。
- 加载 checkpoint 并恢复模型和优化器参数。

---

## 6. 推理与投票机制

- **test_S3DIS.py**
  - 使用 `ModelTester` 类，支持多次投票（num_vote），对每个点多次推理平滑结果。
  - 先在下采样点云上统计混淆矩阵和 IoU，再将预测结果投影回原始点云，输出最终混淆矩阵和 IoU。
  - 输出预测结果为 ply 文件，便于可视化。
- **test_SemanticKITTI.py**
  - 直接在每帧点云上进行推理和投票，直到所有点的“possibility”大于阈值。
  - 支持生成伪标签（gen_pseudo），可用于半监督或自训练。
  - 结果保存为 label 文件或 npy 文件，便于官方评测脚本使用。
  - 支持对验证集计算混淆矩阵、IoU 和准确率。

---

## 7. 标签映射与输出格式

- **test_S3DIS.py**
  - 标签为 13 类，直接输出预测和标签。
  - 输出 ply 文件（`val_preds/*.ply`）。
- **test_SemanticKITTI.py**
  - 标签为 20 类，需根据 yaml 文件做标签 remap。
  - 输出 label 文件（`predictions/*.label`）或 npy 文件，兼容官方评测。

---

## 8. 其它细节

- **test_S3DIS.py**
  - 日志文件名包含测试 Area 信息（如 `log_test_Area_5.txt`）。
  - 只运行一次 epoch，适合单次评估。
- **test_SemanticKITTI.py**
  - 日志文件名固定为 `log_test_kitti.txt`。
  - 支持多 epoch 投票，直到所有点被充分预测。
  - 支持伪标签生成和半监督流程。

---

## 总结表

| 脚本名              | 针对数据集      | 配置/数据集类         | 标签类别 | 输出格式         | 投票机制 | 支持伪标签 | 日志文件名           |
|---------------------|----------------|-----------------------|----------|------------------|----------|------------|----------------------|
| test_S3DIS.py       | S3DIS          | ConfigS3DIS/S3DIS     | 13       | ply              | 支持     | 否         | log_test_Area_{n}.txt|
| test_SemanticKITTI.py| SemanticKITTI | ConfigSemanticKITTI/SemanticKITTI | 20 | label/npy        | 支持     | 支持       | log_test_kitti.txt   |

---

## 一句话总结

**test_S3DIS.py 和 test_SemanticKITTI.py 都是 RandLA-Net 的测试脚本，但分别针对室内和自动驾驶点云分割数据集，数据加载、标签处理、输出格式和功能细节均做了
