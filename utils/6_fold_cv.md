# 6_fold_cv.py文件解析
## 文件详解
- 功能：对预测点云结果文件（.ply）进行逐文件评估，与原始点云标签进行对比，计算整体精度（accuracy）、每一类别的IoU、mean IoU、每类别的accuracy、mean accuracy，并可视化点云
- 这段代码是一个**点云语义分割结果评估脚本**，针对S3DIS数据集的分割任务，主要用于评估模型预测结果的精度（accuracy）和类别平均交并比（mean IoU, mIoU）、类别平均精度（mean accuracy, mAcc）。脚本会遍历预测点云（ply文件），与原始点云的label进行比对，统计评估指标。

---
## 代码详解
### 一、预备环节
```python
import numpy as np   
import glob, os, sys
# 导入numpy用于数值计算，glob用于文件路径匹配，os和sys用于路径操作。

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
# 获取当前脚本文件的目录(BASE_DIR)和项目根目录(ROOT_DIR)，并将根目录加到sys.path，方便import自定义模块。

from helper_ply import read_ply
from helper_tool import Plot
# 导入自定义的点云读取函数和可视化工具类。
```
---

### 二、主体函数main
-输入：
    - 预测结果目录（dase_dir)：存储模型预测结果的.ply文件，每个文件包含预测的类别标签（pred字段）
    - 原始数据目录（original_data_dir）：存储原始点云数据的.ply文件，包含了点坐标（x，y，z）、RGB颜色、真实类别标签（class）
    - 类别数量（13）：使用的是S3DIS数据集（斯坦福3D室内场景数据集），默认有13个语义类别
- 输出：
    - 逐文件准确率：对每个.ply文件，计算并打印七分类准确率
    - 整体准确率（eval accuracy）：所有测试文件的平均分类准确率
    - 各类别IoU（交并比）：每个类别的IoU值
    - 平均IoU（mIoU）：所有类别IoU的平均值
    - 各类别准确率：每个类别的准确率（acc_list）
    - 平均准确率（mAcc）：所有类别准确率的平均值，反映模型在每个类别上的平均分类能力

```python
if __name__ == '__main__':
    base_dir = '/data/liuxuexun/dataset/S3DIS/myresult'  # 预测结果目录
    original_data_dir = '/data/liuxuexun/dataset/S3DIS/original_ply'  # 原始数据目录

    data_path = glob.glob(os.path.join(base_dir, '*.ply'))
    data_path = np.sort(data_path)
    # 读取base_dir下所有ply文件路径，并排序（保证每次遍历顺序一致）。

    test_total_correct = 0
    test_total_seen = 0
    gt_classes = [0 for _ in range(13)]  # 每类真实点数
    positive_classes = [0 for _ in range(13)]  # 每类被预测为该类的点数
    true_positive_classes = [0 for _ in range(13)]  # 每类真正预测正确的点数
    visualization = False
    # 初始化全局统计量。13是类别数（S3DIS默认13类）。

    for file_name in data_path:  # 遍历所有预测ply文件，对每一个文件进行评估
        pred_data = read_ply(file_name)
        pred = pred_data['pred']
        # 读取预测结果的ply文件，获得预测标签（通常是pred字段）。
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-4] + '.ply'))
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T
        # 读取同名原始点云文件，提取真实标签(class字段)和点坐标（x, y, z）。

        ##################
        # Visualize data #      可视化（如果需要）
        ##################
        if visualization:  # 判断是否开启可视化模式。如果visualization == True，则执行可视化相关代码,visualization是一个布尔变量。
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            # 将每个点的RGB颜色通道堆叠成一个二维数组，每行为一个点的RGB值。
            # original_data['red']：所有点的红色通道值（一维数组）。
            # original_data['green']：所有点的绿色通道值（一维数组）。
            # original_data['blue']：所有点的蓝色通道值（一维数组）。
            # np.vstack(...)：将三个一维数组按行堆叠，得到形状为(3, N)的数组。
            # .T：转置为(N, 3)数组，每一行代表一个点的RGB颜色。
            xyzrgb = np.concatenate([points, colors], axis=-1)
            # 将点的3D坐标和颜色拼接成一个(N, 6)数组，每行格式为[x, y, z, r, g, b]。
            # points：点的三维坐标，形状为(N, 3)，每行[x, y, z]。
            # colors：点的RGB颜色，形状为(N, 3)，每行[r, g, b]。
            # axis=-1：按最后一个维度拼接，即行不变，列合并。
            Plot.draw_pc(xyzrgb)  # visualize raw point clouds  可视化原始点云，颜色根据RGB显示。
            # 通常会弹出一个三维窗口，显示带颜色的点云。
            Plot.draw_pc_sem_ins(points, labels)  # visualize ground-truth  以类别标签对点云进行可视化（显示真实标签）。
            # points：点的坐标，形状为(N, 3)。labels：每个点的真实语义类别标签，形状为(N, )。
            Plot.draw_pc_sem_ins(points, pred)  # visualize prediction
            # 以类别标签对点云进行可视化（显示预测标签）。points：点的坐标，形状为(N, 3)。pred：每个点的预测语义类别标签，形状为(N, )。

        correct = np.sum(pred == labels)  # 预测正确的位置为True，否则为False。
        # 统计当前点云文件中预测正确的点的数量。
        # pred：一个一维数组，表示当前文件中每个点的预测类别（通常为整数，范围0~12）。
        # labels：一个一维数组，表示当前文件中每个点的真实类别（ground truth）。
        # np.sum(...)：将布尔型数组中为True的数量加总，得到预测正确的点的总数。
        print(str(file_name.split('/')[-1][:-4]) + '_acc:' + str(correct / float(len(labels))))
        # 输出当前点云文件的名称以及该文件上的分类准确率。
        # file_name：当前正在处理的点云文件的完整路径。
        # file_name.split('/')[-1]：提取文件名（如Area_1_office_1.ply）。
        # [:-4]：去掉文件名的后缀“.ply”，只保留主名。
        # correct：上一行统计的预测正确点数。
        # len(labels)：该点云文件中点的总数。
        test_total_correct += correct  # test_total_correct：一个全局变量，记录所有文件中预测正确点的总数。
        test_total_seen += len(labels)  # test_total_seen：一个全局变量，记录所有文件的点的总数。
        # 累计全体文件预测正确的总点数和总点数，为整体准确率的计算做准备。
        # correct：本文件预测正确的点数。
        # len(labels)：本文件所有点的数量。

        for j in range(len(labels)):  # 遍历当前点云文件中每一个点。
        # labels：当前点云文件中每个点的真实类别标签（ground truth），是一个一维数组。
        # range(len(labels))：产生一个索引序列，覆盖所有点。
            gt_l = int(labels[j])   # 获取当前第j个点的真实类别标签，并转换为整数类型。labels[j]：第j个点的真实类别（例如0~12）。gt_l：当前点的真实类别标签。
            pred_l = int(pred[j])   # 获取当前第j个点的预测类别标签，并转换为整数类型。pred[j]：第j个点的模型预测类别（例如0~12）。pred_l：当前点的预测类别标签。
            gt_classes[gt_l] += 1  # 累计真实属于类别gt_l的点的数量。gt_classes：长度为类别数（如13）的列表，每个位置记录对应类别的真实点数。gt_l：当前点的真实类别索引。
            positive_classes[pred_l] += 1  # 累计被模型预测为类别pred_l的点的数量。
            # positive_classes：长度为类别数的列表，每个位置记录被预测为该类别的点数。
            # pred_l：当前点的预测类别索引。
            true_positive_classes[gt_l] += int(gt_l == pred_l)  # 如果本点预测正确（预测类别等于真实类别），则该类别的真正数加1。
            # true_positive_classes：长度为类别数的列表，每个位置记录真实为该类别且预测为该类别的点数。
            # gt_l == pred_l：判断是否预测正确，若正确为True（即1），否则为False（即0）。

    iou_list = [] # 初始化一个空列表，用于存储每个类的IoU值
    for n in range(13):  #循环遍历13个类别
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        # 计算第n个类别的IoU值
        # true_positive_classes[n]：第n个类别的真正例（TP），即预测正确且属于该类别的像素数
        # gt_classes[n]：第n个类别的真实标签数
        # positive_classes[n]：第n个类别的预测为正例的像素数（Pred）
        iou_list.append(iou)
        # 将当前类别的IoU值添加到iou_list中
    mean_iou = sum(iou_list) / 13.0
    # 计算所有类别的平均IoU（mIoU）
    print('eval accuracy: {}'.format(test_total_correct / float(test_total_seen)))
    # 打印整体准确率：test_total_correct：所有类别中预测正确的像素总数
    # test_total_seen：所有像素的总数
    print('mean IOU:{}'.format(mean_iou))
    # 打印平均IoU（mIoU）
    print(iou_list)
    # 打印每个类别的IoU值列表

    acc_list = []
    for n in range(13):  # 再次循环13个类别
        acc = true_positive_classes[n] / float(gt_classes[n]) # 计算第n个类别的准确率
        # true_positive_classes[n]：第n个类别的真正例（TP）
        # gt_classes[n]：第n个类别的真实标签数（GT）
        acc_list.append(acc)  # 将当前类别的准确率添加到acc_list中
    mean_acc = sum(acc_list) / 13.0  # 计算平均准确率（mAcc），即acc_list中所有值的平均值
    print('mAcc value is :{}'.format(mean_acc))  # 打印平均准确率（mAcc）
```
