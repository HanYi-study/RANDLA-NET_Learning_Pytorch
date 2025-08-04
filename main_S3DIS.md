# main_S3DIS.py文件解析
---
## 尚未解决的问题：  
1. 如果选择Area5用作测试集，那为什么训练的结果中生成的日志只有Area5，并且测试结果中也是只有Area5。是否哪里的设定有问题？
2. 什么是cfg？好像是一个文件记载了配置数据。
3. 优化器Adam详细了解，包含了哪些参数
4. 学习率的大小对模型收敛程度的影响是什么？
---
## 主体功能概览：  
该脚本是 RandLA-Net 在 S3DIS 数据集上的训练与验证主程序，主要完成：

- S3DIS 数据加载与训练/验证集划分（Leave-One-Out）
- 网络结构初始化与 checkpoint 加载
- 完整训练 + 评估循环（记录最优 mIoU）
- 日志与模型保存功能

---

##  主体模块分解：

### 一、 Argument 参数解析

```python
parser = argparse.ArgumentParser()
# 主要控制训练设备、最大轮数、保存路径、验证区域（area），用于从命令好接受参数输入  
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
# 指定模型训练过程中保存的 checkpoint文件路径。如果填了该路径，训练将从该 checkpoint 恢复（即断点续训）。default是默认值。
parser.add_argument('--log_dir', default='train_output', help='Dump dir to save model checkpoint [default: log]')
# 指定日志和模型保存文件夹的路径，其中包括训练日志文件、模型权重文件等。默认train_output（该文件夹在文件结构中可见，每执行一次该文件夹下就会生成一次执行日志以及模型权重文件，模型权重文件可用于test测试使用）
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
# 指定训练的 最大 epoch 数，即训练将循环多少轮。默认100轮，可以通过train_output中7-12生成的训练日志看出，默认了100个epoch（0-99epoch）。
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 0], -1 for cpu')
# 指定训练使用哪一块GPU，默认使用0号GPU，若该值设置为-1，则使用cpu进行训练
parser.add_argument('--test_area', type=int, default=5,help='Which area to use for test (others use to train), option: 1-6 [default: 5]')
# 指定选用数据集中哪个区域用作测试集，剩下几个都用于训练。S3DIS数据集中共有6个Area（区域），默认第五个区域用作测试集。是一种Leave-One-Out（留一法）交叉验证。
FLAGS = parser.parse_args()
# 将上述所有定义的命令行参数解析成一个对象FLAGS，其属性：FLAGS.checkpoint_path  # 访问模型路径/FLAGS.log_dir # 访问日志输出目录/FLAGS.max_epoch # 最大训练轮数/FLAGS.gpu # 使用GPU编号/FLAGS.test_area # S3DIS中测试区域编号

```

- `--checkpoint_path`: 恢复训练时用的模型路径（string类型）
- `--log_dir`: 日志保存文件夹（含时间戳）（string类型）
- `--max_epoch`: 最大训练轮数（默认100）（int类型）
- `--gpu`: 使用哪块 GPU（如 0）
- `--test_area`: 留出验证区域（S3DIS Area1~6）

---

### 二、 日志系统初始化

```python
#首先：创建日志目录与文件路径
LOG_DIR = FLAGS.log_dir
# 从命令行参数中读取对象FLAGS中日志属性log_dir的输出目录（默认是train_output），可以输入从而实现将运行日志和权重文件存放进指定目录
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
# 将LOG_DIR改成带有当前时间戳的子目录
# 使用time.gmtime() 获取当前 GMT 时间（即格林威治标准时间）
# time.strftime('%Y-%m-%d_%H-%M-%S', ...) 格式化为日期字符串
# 训练结束后生成的日志和模型权重文件所在的文件夹名称如下：2025-07-12 01-48-28
# 可确保每次运行训练时日志不会覆盖旧的，而是保存在不同的带时间戳的子文件夹中。
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
# 如果上面拼接得到的路径还不存在，就自动创建该多级目录。
# 保证后续文件保存不会因目录缺失报错。

#其次：创建日志文件路径并打开文件
log_file_name = f'log_train_Area_{FLAGS.test_area:d}.txt'
# 构造日志文件名，表示哪个测试区域的训练日志。
# 经过训练之后创建的文件名如下：log_train_Area_5.txt
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')
# 在刚才创建的日志目录中，以追加写入（'a'）模式打开这个日志文件。
# 如果中途断点训练，日志文件仍能接着写，不会覆盖。也可以在训练中动态不断向日志文件写入内容。

#最后：日志写入函数
def log_string(out_str):
# 这是一个封装函数，用于在训练过程中统一输出日志信息。
# 每次调用log_string("某条日志")，就会：1.写入.txt文件 2.打印到终端
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

```

- 自动生成带时间戳的训练日志目录
- `log_string()` 函数用于写日志并打印信息
- 输出日志如：`train_output/2025-07-12_01-48-28/log_train_Area_5.txt`

---

### 三、 数据集加载与划分

```python
# 第一步：创建数据集对象
dataset = S3DIS(FLAGS.test_area)
# 初始化S3DIS数据集类对象
# FLAGS.test_area：指定用于测试的区域
# S3DIS 是一个封装的数据类（通常定义在 s3dis_dataset.py）：读取预处理过的 .txt、.label 文件/自动划分训练和验证集（使用留一验证策略）/保存所有点云坐标、颜色、标签等原始信息

# 第二步：创建训练与验证采样器
training_dataset = S3DISSampler(dataset, 'training')
validation_dataset = S3DISSampler(dataset, 'validation')
# S3DISSampler 是用于采样点云块的封装数据集接口类，封装了 RandLA-Net 的采样逻辑。
# 第一个参数：dataset 是前面构造的完整 S3DIS 数据集对象
# 第二个参数：'training'：使用训练区域的数据/'validation'：使用测试区域的数据（Area i）
# 对点云块进行随机采样，实现 RandLA-Net 所需的 固定点数采样
# 返回构造好的训练数据 batch，包括：xyz 坐标、colors、labels、neighborhood indices 等张量/collate_fn 用于自定义 batch 打包方式。

# 第三步：构建PyTorch 数据加载器（DataLoader）
training_dataloader = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True,collate_fn=training_dataset.collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.val_batch_size, shuffle=True,collate_fn=validation_dataset.collate_fn)
# 为训练数据构造批处理数据流 training_dataloader/为测试数据构造批处理数据流 validation_dataloader
# training_dataset / validation_dataset：上一步中构造的采样器数据集 
# batch_size	每个 batch 包含的样本数量，训练集的样本数量来自配置 cfg.batch_size（如 6），测试集的样本数量来自配置cfg.val_batch_size（即batch大小为cfg.val_batch_size）
# shuffle=True	每个 epoch 随机打乱训练样本
# collate_fn	自定义批处理打包方法，确保 RandLA-Net 能正确读取每一批点云

# 第四步：打印数据加载器中batch的数量，先输出的是训练集batch数量，再输出测试集的batch数量 
print(len(training_dataloader), len(validation_dataloader))
# 打印训练和验证集中总的 batch 数量（即 epoch 中的迭代次数）
# 若输出是620 156 / 训练集被分为620个batch（每轮训练将迭代620次） / 验证集被分为 156 个 batch（每轮验证将迭代 156 次）
```

- 加载完整 S3DIS 点云场景数据（包含标签）
- 将除指定 Area 的其余区域作为训练集
- 封装成 `DataLoader`（带 batch、随机打乱、collate_fn）

```text
流程图（简单版）：
S3DIS Dataset        
     ↓
S3DISSampler (训练/验证采样)
     ↓
DataLoader (训练/验证数据加载器)
     ↓
生成 batched 点云数据用于网络输入
```


####  输出变量：
- `training_dataloader` 中的 batch 数（即训练集中一共有多少个 batch）
- `validation_dataloader`中的 batch 数（即验证集中一共有多少个 batch）

---

### 四、 网络构建与加载

```python
# 第一步：选择设备
if FLAGS.gpu >= 0:
# 判断是否指定使用 GPU（可以通过通过命令行参数传入 --gpu=0 或其他大于等于 0 的值，即FLAGS.gpu传入gpu参数）
    if torch.cuda.is_available():
    # 如果系统支持 CUDA（即存在 GPU 并安装了 CUDA 驱动）
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
        # 使用指定编号的 GPU，转换为 torch.device 对象
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        # 如果系统没有可用gpu，则会给出警告
        FLAGS.gpu = torch.device('cpu')
        # 强制使用cpu
else:
# 若一开始给定的参数就是-1，则直接使用cpu
    FLAGS.gpu = torch.device('cpu')
device = FLAGS.gpu
# 将设备对象赋值给 device，后续统一使用 device 来迁移网络、数据等，便于管理

# 第二步：初始化网络结构并迁移到指定设备
net = Network(cfg)
# 实例化 RandLA-Net 网络结构对象，cfg 包含网络参数配置（通道数、层数、类别数等）。
net.to(device)
# 将模型加载到目标设备上（GPU 或 CPU），否则模型无法参与训练。

# 第三步：初始化优化器
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
# 使用 Adam 优化器
# net.parameters()：待优化的参数
# cfg.learning_rate：学习率，来自配置文件

# 第四步：初始化相关状态变量
it = -1
# 训练轮数初始化为-1
# it通常用于学习率调度器
# it也用于批归一化调度器等
start_epoch = 0
# 训练起始的epoch是0
# 默认为0，若之后加载了checkpoint，将被覆盖

# 第五步：加载模型断点（checkpoint）
CHECKPOINT_PATH = FLAGS.checkpoint_path
# 从命令行读取模型断点路径（如果有则读取0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
# 判断是否传入了合法的chechpoint文件路径
    checkpoint = torch.load(CHECKPOINT_PATH)
    # 加载 checkpoint 内容，是一个字典对象，通常包含如下几点：
    # 1. model_state_dict:模型参数
    # 2. optimizer_state_dict:优化器状态
    # 3. epoch:已训练的轮数 / 或者还包含其他内容
    net.load_state_dict(checkpoint['model_state_dict'])
    # 恢复网络权重参数，继续训练
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 恢复优化器状态，例如历史动量、学习率等
    start_epoch = checkpoint['epoch']
    # 设置训练开始的epoch为断点文件中记录的值，继续从epoch开始。
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
    # 打印日志，提示成功加载断点模型
#if下的所有操作都是在有断点的前提下执行的，如果没有断点则一概不执行

```

- 初始化 RandLA-Net 模型
- 将模型移动到 CUDA 或 CPU 上
- 若提供 checkpoint 路径，则恢复模型 + 优化器状态

---

### 五、 学习率调整函数（训练函数1）

```python
def adjust_learning_rate(optimizer, epoch):
# 根据当前的 epoch，使用预设的衰减系数对优化器中的学习率 lr 进行调整
# 训练过程中逐渐降低学习率，提升模型收敛稳定性
# optimizer: 当前使用的优化器（如 Adam）
# epoch: 当前训练轮次（从 0 开始）
    lr = optimizer.param_groups[0]['lr']
    # 从优化器中获取当前学习率lr
    # optimizer.param_groups 是一个长度为1的列表（可能有时不为一），列表里面是字典，字典中有该优化器相关的参数，通常只包含一个字典项（除非你为不同层设置了不同学习率）
    # param_groups[0]['lr'] 即当前正在使用的学习率值
    lr = lr * cfg.lr_decays[epoch]
    # cfg.lr_decays一个有500个键值对的字典，为每一个 epoch 预定义了一个衰减因子（如 0.95），每个键对应的值都是0.95（衰减因子），也就是每个epoch学习率衰减0.95
    # 每轮训练，学习率都乘上该轮对应的衰减系数
    # 实现了逐 epoch 衰减学习率策略
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 遍历优化器中所有参数组（通常就1个）
    # 将新计算出的学习率lr赋值回优化器
    # 完成实际的学习率更新操作，使下一轮训练以新的学习率进行
```

- 每个 epoch 学习率乘以 `cfg.lr_decays[epoch]`
- 通常是指数衰减，如 0.95^epoch
- 最终效果：每训练一个 epoch，就将当前学习率乘上一个衰减系数（如 0.95），如果初始为 0.01，第 1 轮后变为 0.0095，第 2 轮后变为 0.009025，依此类推

---

### 六、 训练单个 epoch 函数：`train_one_epoch()` （训练函数2）

```python
def train_one_epoch():
# 训练一个epoch（完整的一轮遍历训练集）
# 定义训练过程的主函数，执行一轮完整的 forward + backward 过程
# 通常由 train() 主函数调用
    stat_dict = {}  # collect statistics
    # 用于记录训练过程中每个 batch 的 loss、accuracy、IoU 等统计量
    # 最终用于日志输出
    adjust_learning_rate(optimizer, EPOCH_CNT)
    # 调用函数-----------------------------------------------------------------------------------------------------------------------------------------------------------08/04
    net.train()  # set model to training mode
    iou_calc = IoUCalculator(cfg)  # 初始化IOU计算器
    for batch_idx, batch_data in enumerate(training_dataloader):
        t_start = time.time()
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg, device)
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)  # 保存训练结果，用于计算iou

        # Accumulate statistics and print out           # 累计损失和准确率
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 50  # 本来是10
        if (batch_idx + 1) % batch_interval == 0:
            t_end = time.time()
            # log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            # # TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
            # #     (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            # for key in sorted(stat_dict.keys()):
            #     log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
            #     stat_dict[key] = 0
            log_string('Step %03d Loss %.3f Acc %.2f lr %.5f --- %.2f ms/batch' % (batch_idx + 1,
                                                                                   stat_dict['loss'] / batch_interval,
                                                                                   stat_dict['acc'] / batch_interval,
                                                                                   optimizer.param_groups[0]['lr'],
                                                                                   1000 * (t_end - t_start)))
            stat_dict['loss'], stat_dict['acc'] = 0, 0
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)

```

#### 功能：
- 训练状态设为 `net.train()`
- 遍历所有训练 batch：
  - 输入数据送入模型，输出 `end_points`
  - 计算 `loss` → 反向传播 → 优化器更新
  - 累加 `loss`、`acc`、添加到 IoU 计算器
- 每 50 个 batch 输出一次训练统计信息
- 最后输出整个 epoch 的 `mean IoU` 和 per-class `IoU`

#### 输入：
- `training_dataloader`, `net`, `optimizer`, `cfg`, `device`

---

### 七、 评估函数：`evaluate_one_epoch()`（训练函数3）

```python
def evaluate_one_epoch():
```

#### 功能：
- 设置为 `net.eval()`（禁用 Dropout 和 BN 更新）
- 只进行前向传播，不进行反向传播
- 统计验证集上的 `loss`、`acc`、`IoU`

#### 输出：
- 返回 `mean_iou`（float）

---

### 八、 训练主控函数：`train(start_epoch)`（训练函数4）

```python
def train(start_epoch):
```

#### 功能：
- 控制整个训练流程：
  - 逐轮训练 (`train_one_epoch`)
  - 每轮验证 (`evaluate_one_epoch`)
  - 保存最优模型 checkpoint（按 mIoU 选）

#### 输出：
- 日志写入
- 模型保存到：`checkpoint.tar`

---

##  脚本主入口

```python
if __name__ == '__main__':
    train(start_epoch)
```

- 以指定 `start_epoch`（默认0）开始训练

---

##  关键数据结构

| 变量名           | 类型            | 描述                         |
|------------------|-----------------|------------------------------|
| `batch_data`     | dict[list/torch] | 每个 batch 的点云块、标签等   |
| `end_points`     | dict             | 网络输出（loss、acc、预测标签） |
| `iou_list`       | list[float]      | 每类的 IoU 值                |
| `mean_iou`       | float            | 所有类别的平均 IoU           |

---

##  核心流程图（文字版）

```text
[ 参数解析 ] 
     ↓
[ 创建日志文件夹 ]
     ↓
[ 加载 S3DIS 数据集 ]
     ↓
[ 初始化模型（可加载 checkpoint） ]
     ↓
for epoch in max_epoch:
    ├─> [训练 train_one_epoch()]
    ├─> [评估 evaluate_one_epoch()]
    └─> [保存最佳模型（mIoU最高）]
```

---

##  总结一句话

该脚本是 RandLA-Net 在 S3DIS 上的完整训练框架，包括 **数据准备、网络构建、训练与验证流程、性能评估与保存**。你可将其作为主控脚本串联所有流程模块。

---
