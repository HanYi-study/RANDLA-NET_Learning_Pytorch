# data_prepare_s3dis.py文件解析
## 文件详解

主要功能：
- 将原始 S3DIS 数据集的每个房间的点云（txt格式）合并为一个带标签的 ply 文件
    - 每个房间的所有点（XYZRGBL）合并，标签按类别编号映射。
    - 输出到 original_ply 文件夹。
- 对每个房间的点云进行网格下采样（grid_sub_sampling）
    - 生成稀疏点云（减少点数），并归一化颜色。
    - 输出到 input_0.040 文件夹（0.04 为采样网格大小）。
- 为下采样点云建立 KDTree 并保存
    - 便于后续快速空间检索。
- 保存原始点到下采样点的最近邻投影索引和标签
    - 便于后续训练时标签映射。

 主要流程：
 1. 读取每个房间的所有 txt 标注文件，合并为一个大点云，标签编号化。
 2. 坐标归一化（减去最小值，保证正数）。
 3. 保存为 ply 格式。
 4. 下采样点云，保存下采样 ply。
 5. 建立 KDTree，保存为 pkl。
 6. 保存原始点到下采样点的投影索引和标签为 pkl。



模块一）导入依赖和路径设置（环境准备模块）  
- 导入各种依赖库（如numpy、pandas、sklearn、os等）。
- 设置当前文件、上一级目录路径，并把它们加入sys.path，方便后续自定义包的引入。
- 引入点云写入和数据处理相关的自定义工具。
模块二）数据路径与类别信息配置（全局配置模块）
- 设置数据集路径和相关文件夹。
- 读取房间（annotation）路径、类别名称，并建立类别名称到标签的映射。
- 配置点云网格采样参数，准备输出目录。
模块三）点云转换及采样处理函数（核心处理模块）
- 输入一个房间的 annotation 路径，将其中所有点合并成一个点云文件（.ply），并保存。
- 对点云进行下采样，保存采样后的子点云和对应的 KDTree（用于快速空间搜索）。
- 保存原始点到采样点的映射关系（proj_idx），以便后续训练或评估时恢复标签。
模块四）主程序入口及批量处理（批处理模块）
- 遍历所有房间的 annotation 路径，依次调用 convert_pc2ply 处理每一个房间数据。
- 自动化地将原始 txt 注释文件批量转换成 .ply 点云文件，并生成采样、KDTree、映射等文件。
模块五）工具依赖（自定义工具辅助模块）  
- 利用自定义工具进行点云写入（.ply）和点云数据的采样、处理等操作。
- 这些工具实现了标准点云数据格式的读写、空间采样等底层细节。

---

## 模块详解
### 一、导入依赖和路径设置（环境准备模块）
```python
from sklearn.neighbors import KDTree  # 导入 KDTree，用于后续点云的空间索引和最近邻搜索。
from os.path import join, exists, dirname, abspath  # 导入常用的路径操作函数：拼接路径、判断文件是否存在、获取目录名、获取绝对路径。
import numpy as np  # 导入 numpy，进行数值计算和数组操作。
import pandas as pd  # 导入 pandas，主要用于读取 txt 格式的点云数据（如 pd.read_csv）。
import os, sys, glob, pickle
# 导入 os：文件和目录操作。  导入 sys：操作 Python 运行环境（如修改 sys.path）。
# 导入 glob：用于查找符合特定规则的文件路径名。  导入 pickle：用于对象的序列化和反序列化（如保存 KDTree、索引等）。

BASE_DIR = dirname(abspath(__file__))   # 当前文件夹绝对路径
# 获取当前脚本文件的绝对路径，再取其所在目录，赋值给 BASE_DIR。
ROOT_DIR = dirname(BASE_DIR)            # 上一级文件夹路径
# 获取 BASE_DIR 的上一级目录路径，赋值给 ROOT_DIR。
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)               # 将目录添加到系统路径之后就可以添加自己的包
# 把 BASE_DIR 和 ROOT_DIR 加入 Python 的模块搜索路径，方便后续导入同目录或上级目录下的自定义模块。
from helper_ply import write_ply        # 从 helper_ply.py 导入 write_ply 函数，用于将点云数据写入 ply 文件。
from helper_tool import DataProcessing as DP  # 从 helper_tool.py 导入 DataProcessing 类，并重命名为 DP，后续用于点云的下采样等处理。
```

---

### 二、数据路径与类别信息配置（全局配置模块）
```python
dataset_path = '/home/hy/projects/new_project/RandLA-Net-Pytorch-New/data/Stanford3dDataset_v1.2_Aligned_Version'     # 我的路径
#  S3DIS 原始数据集的根目录路径。这里用的是你本地的实际路径。
anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'meta/anno_paths.txt'))]      # 返回一个列表,列表中的每个元素是该文件每一个行
# 读取 meta/anno_paths.txt 文件，每一行是一个房间的相对路径，去掉行尾换行符，得到一个列表。 例如：Area_1/office_2/Annotations。
anno_paths = [join(dataset_path, p) for p in anno_paths]                                  # 将数据集路径和txt文件中的路径结合
# 把上面得到的每个相对路径拼接到 dataset_path 后，得到每个房间的标注文件夹的绝对路径。
# 例如：/home/hy/projects/new_project/RandLA-Net-Pytorch-New/data/Stanford3dDataset_v1.2_Aligned_Version/Area_1/office_2/Annotations

gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'meta/class_names.txt'))]             # 将类别文件中的内容保存在一个列表中
# 读取 meta/class_names.txt，每一行是一个类别名，去掉换行符，得到类别名列表。例如：['ceiling', 'floor', 'wall', ...]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}
# 将类别和对应的下标(索引)以键值对的方式存放到字典中，这里类别对应的序号需要与s3dis_dataset文件中类序号的定义一样
# 把类别名和类别编号（索引）组成字典，便于后续将类别名转为数字标签。例如：{'ceiling': 0, 'floor': 1, ...}

sub_grid_size = 0.04                                                                      # 网格采样时网格的边长
# 下采样时的网格大小（单位：米），即每 4cm 采一个点，控制下采样稀疏程度。
original_pc_folder = join(dirname(dataset_path), 'original_ply')                          # 方便下面在数据文件夹中创建一个ply目录
# original_pc_folder：原始点云（合并并加标签后）的 ply 文件保存目录。路径为数据集目录的上一级下的 original_ply 文件夹。
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))         # 方便下面在数据文件夹中创建目录保存网格采样后的数据
# sub_pc_folder：下采样后点云的 ply 文件保存目录。
# 路径为数据集目录的上一级下的 input_0.040 文件夹（0.04 为采样网格大小）。
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None                  # 创建对应的文件夹
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
# 如果 original_pc_folder 或 sub_pc_folder 不存在，则创建它们。
out_format = '.ply'  # 输出文件格式为 ply（点云常用格式）。
```
- original_pc_folder：保存每个房间合并后的原始点云（带标签），用于备份和可视化。
- sub_pc_folder：保存每个房间下采样后的点云、KDTree、投影索引等，供训练和快速检索使用。

---

### 三、点云转换及采样处理函数（核心处理模块）
- **输入**：anno_path（房间标注目录），save_path（原始 ply 保存路径）
- **输出**：
  - 合并后的原始点云 ply 文件（XYZRGBL）
  - 下采样点云 ply 文件（input_0.040/xxx.ply）
  - KDTree 文件（input_0.040/xxx_KDTree.pkl）
  - 投影索引和标签文件（input_0.040/xxx_proj.pkl）
```python
def convert_pc2ply(anno_path, save_path):
# 该函数将 S3DIS 原始房间标注文件夹（txt）合并为一个带标签的 ply 点云文件，并生成下采样点云、KDTree、投影索引等训练所需的辅助文件。

    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    data_list = [] # 用于存储该房间所有物体的点云及标签。

    for f in glob.glob(join(anno_path, '*.txt')):  # 遍历该房间所有物体的 txt 文件。
        class_name = os.path.basename(f).split('_')[0]                      # 获取类别名，忽略同一类别下的第几个物体
        # 通过文件名获取类别名（如 `chair_1.txt` → `chair`）。     
        if class_name not in gt_class:  # note: in some room there is 'staris' class..  若类别名不在已知类别列表，则归为clutter
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values      # 将txt文件内容读取进来,读取 txt 文件内容为 numpy 数组（每行为 XYZRGB）。
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]     # n行1列矩阵，其中n是这个txt文件的行数，生成点的标签矩阵/为该物体所有点生成类别标签（数字），shape 为 (点数, 1)。
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7            # 将点的标签矩阵和点的特征矩阵合并    点云和标签拼接，加入 `data_list`。

    pc_label = np.concatenate(data_list, 0)                                 # 将列表中的矩阵堆积起来形成一个矩阵,合并所有物体点云为一个大矩阵（N,7）。                          
    xyz_min = np.amin(pc_label, axis=0)[0:3]                                # 取出这个room中所有数据中xyz的最小值
    pc_label[:, 0:3] -= xyz_min                                             # 进行坐标中心偏移所有点的坐标值减去最小值，确保所有坐标都是正数/所有点的 xyz 坐标减去最小值，实现坐标归一化（保证正数，便于后续处理）。

    xyz = pc_label[:, :3].astype(np.float32)                                # 从矩阵截取部分所需数据
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    # 分别提取 xyz、rgb、label。
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    # 保存为 ply 文件，字段为 x, y, z, red, green, blue, class。

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)  # 进行网格采样
     # 对点云进行网格下采样（如 0.04m），得到稀疏点云。
    sub_colors = sub_colors / 255.0                                                             # 颜色归一化,颜色归一化到 [0,1]。
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')    # 保存下采样点云为 ply 文件，存放在 `input_0.040/` 目录。
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])    # 不再保存在savepath下了，而是保存在input0.04的文件夹下

    search_tree = KDTree(sub_xyz)  # 用下采样点云的 xyz 建立 KDTree，便于后续空间检索。
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')   # 保存 KDTree 对象为 pkl 文件。
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))        # 从子云中查询离原始云xyz最近的点的索引列表（投影关系），返回的维度为(xyz的长度,1)
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')  # 保存最近邻索引和标签为 pkl 文件，便于训练时标签映射。
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)
```

---

### 四、主程序入口及批量处理（批处理模块）
- 输入：`anno_paths` 是所有房间标注文件夹的路径列表，作为输入。
- 经过了什么处理：（对每个房间）
    - 合并所有txt 点云文件，生成带标签的 ply 文件（保存在 `original_ply/`）。
    - 进行下采样，生成稀疏点云 ply 文件（保存在 `input_0.040/`）。
    - 建立 KDTree，保存为 pkl 文件（保存在 `input_0.040/`）。
    - 保存原始点到下采样点的投影索引和标签（保存在 `input_0.040/`）。
- 输出：
    - `original_ply/Area_X_room_Y.ply`：合并后的原始点云（XYZRGBL）。
    - `input_0.040/Area_X_room_Y.ply`：下采样后的点云（XYZRGBL，颜色归一化）。
    - `input_0.040/Area_X_room_Y_KDTree.pkl`：KDTree 对象。
    - `input_0.040/Area_X_room_Y_proj.pkl`：原始点到下采样点的最近邻索引和标签。

```python
if __name__ == '__main__':
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for annotation_path in anno_paths:  # 遍历所有房间的标注文件夹路径（`anno_paths` 是一个列表，每个元素如 `.../Area_1/office_2/Annotations`）。
        print(annotation_path)  # 打印当前正在处理的房间标注文件夹路径，便于跟踪进度。
        elements = str(annotation_path).split('/')  # 将当前标注路径按 `/` 分割成列表。例如 `.../Area_1/office_2/Annotations` → `['...', 'Area_1', 'office_2', 'Annotations']`。
        out_file_name = elements[-3] + '_' + elements[-2] + out_format
        # 取倒数第3和第2个元素（如 `Area_1` 和 `office_2`），拼接成输出文件名（如 `Area_1_office_2.ply`）。
        # `out_format` 是 `.ply`，表示输出为 ply 格式。
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))
        # 调用前面定义的 `convert_pc2ply` 函数，处理当前房间的所有 txt 标注文件。
        #`annotation_path`：输入，当前房间的标注文件夹路径。
        #`join(original_pc_folder, out_file_name)`：输出，合并后原始点云 ply 文件的保存路径（如 `original_ply/Area_1_office_2.ply`）。
```
### 五、工具依赖（自定义工具辅助模块 / 在模块一中出现过）
```python
from helper_ply import write_ply        
from helper_tool import DataProcessing as DP
```
