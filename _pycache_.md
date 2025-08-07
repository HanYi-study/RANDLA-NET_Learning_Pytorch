# `_pycache_`文件夹解析
## 目录结构
```text
RandLA-Net-Pytorch-New/
├── __pycache__/   <-------------它在这里
│   ├── helper_ply.cpython-37.pyc     # 处理 .ply 格式点云文件的读写和操作相关工具函数。
│   ├── helper_tool.cpython-37.pyc    # 各类辅助工具函数，例如数据处理、配置管理、统计等通用代码。
│   ├── helper_tool.cpython-313.pyc   # 各类辅助工具函数，例如数据处理、配置管理、统计等通用代码。
│   ├── pytorch_utils.cpython-37.pyc  # PyTorch 框架相关的辅助代码，如定义卷积层封装、网络模块等。
│   ├── RandLANet.cpython-37.pyc      # RandLA-Net 模型核心代码，定义网络结构和前向传播等。
│   └── s3dis_dataset.cpython-37.pyc  # 处理 S3DIS 数据集相关的加载、采样和预处理。
├── helper_ply.py
├── helper_tool.py
├── pytorch_utils.py
├── RandLANet.py
├── s3dis_dataset.py
└── ...
```
## 其他信息
- 作用：  
__pycache__ 是 Python 自动生成的文件夹，用于**缓存已编译的 Python 字节码文件（.pyc 文件）**。这些文件是 .py 源码经过 Python 解释器编译成的中间字节码，加速下次运行。
- 为什么自动生成：  
当 Python 第一次导入一个模块时，会**自动编译**成 .pyc 字节码，并存放在当前目录下的 __pycache__ 文件夹中。  
这样，下次导入同一个模块时，Python 可以直接加载 .pyc，减少编译时间，**提高运行速度**。
- 文件命名规则：  
.pyc 文件名格式一般为 “ 模块名.cpython-版本号.pyc ”，比如 “ helper_tool.cpython-37.pyc ” 表示 Python 3.7 编译的字节码。
