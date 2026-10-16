# 导入必要的库
import numpy as np  # 用于数值计算和数组操作
import matplotlib.pyplot as plt  # 用于数据可视化
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 关键修复！
plt.rcParams['axes.unicode_minus'] = False

import numpy as np

def analyze_qm_file(file_path):
    """
    分析.qm质量图文件的完整函数
    参数:
        file_path: .qm文件的路径（字符串）
    返回:
        qm_data: 解析后的质量图数据（numpy数组）
    """

    # ==================== 1. 数据读取和解析 ====================
    with open(file_path, 'r') as file:  # 以只读模式打开文件
        lines = file.readlines()  # 读取所有行，返回字符串列表

    # 将文本数据转换为二维数组
    qm_data = []
    for line in lines:  # 遍历每一行
        # 移除首尾空白字符并按空格分割，然后转换为整数列表
        nums = list(map(int, line.strip().split()))
        qm_data.append(nums)  # 将当前行数据添加到列表中

    qm_data = np.array(qm_data)  # 将列表转换为numpy数组

    # 打印基本信息
    print("数据形状:", qm_data.shape)  # 输出数组维度（行数, 列数）
    print("质量分范围:", np.min(qm_data), "~", np.max(qm_data))

    # ==================== 2. 可视化分析 ====================
    plt.figure(figsize=(12, 5))  # 创建图形窗口，设置大小

    # -------------------- 2.1 热力图 --------------------
    plt.subplot(1, 2, 1)  # 创建子图（1行2列的第1个）
    heatmap = plt.imshow(qm_data,
                         cmap='viridis',  # 使用viridis颜色映射
                         interpolation='nearest')  # 禁用插值，保持原始分辨率
    plt.colorbar(heatmap, label='质量分')  # 添加颜色条并设置标签
    plt.title("质量图热力图")  # 设置标题
    plt.xlabel("X坐标")  # X轴标签
    plt.ylabel("Y坐标")  # Y轴标签

    # -------------------- 2.2 统计直方图 --------------------
    plt.subplot(1, 2, 2)  # 创建子图（1行2列的第2个）
    plt.hist(qm_data.flatten(),  # 将二维数组展平为一维
             bins=np.arange(-0.5, 5.5, 1),  # 设置bin范围为-0.5到4.5，间隔1
             edgecolor='black',  # 柱状图边框颜色
             alpha=0.7)  # 设置透明度
    plt.xticks(range(5))  # X轴刻度设置为0-4
    plt.title("质量分分布")  # 设置标题
    plt.xlabel("质量等级")  # X轴标签
    plt.ylabel("频数")  # Y轴标签
    plt.grid(axis='y', alpha=0.3)  # 添加横向网格线（透明度0.3）

    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示图形

    # ==================== 3. 3D可视化 ====================
    fig = plt.figure(figsize=(10, 7))  # 创建新图形窗口
    ax = fig.add_subplot(111, projection='3d')  # 添加3D子图

    # 生成网格坐标
    x = np.arange(qm_data.shape[1])  # X轴坐标（0到列数-1）
    y = np.arange(qm_data.shape[0])  # Y轴坐标（0到行数-1）
    x, y = np.meshgrid(x, y)  # 生成网格坐标矩阵

    # 绘制3D曲面
    surf = ax.plot_surface(x, y, qm_data,
                           cmap='viridis',  # 颜色映射
                           rstride=1,  # 行跨度
                           cstride=1,  # 列跨度
                           alpha=0.8)  # 设置透明度
    fig.colorbar(surf, label='质量分')  # 添加颜色条

    # 设置3D图形属性
    ax.set_title("质量图3D可视化")  # 标题
    ax.set_xlabel("X坐标")  # X轴标签
    ax.set_ylabel("Y坐标")  # Y轴标签
    ax.set_zlabel("质量分")  # Z轴标签
    ax.view_init(elev=30, azim=45)  # 设置视角（仰角30度，方位角45度）

    plt.show()  # 显示图形

    # ==================== 4. 高级分析 ====================
    # 计算各质量等级的占比
    unique, counts = np.unique(qm_data, return_counts=True)  # 统计各数值出现次数
    total = qm_data.size  # 总像素数
    print("\n质量分布统计:")
    for val, cnt in zip(unique, counts):
        print(f"等级{val}: {cnt}像素 ({cnt / total:.1%})")  # 打印百分比

    # 检测高质量区域（等级≥3）
    high_quality = qm_data >= 3  # 生成布尔掩码
    hq_percentage = np.mean(high_quality)  # 计算True的比例
    print(f"\n高质量区域占比: {hq_percentage:.1%}")

    return qm_data  # 返回解析后的数据


# 示例调用（替换为实际文件路径）
if __name__ == "__main__":
    file_path = "26_3_features.qm"  # 假设的文件路径
    qm_data = analyze_qm_file(file_path)  # 调用分析函数
"""
1. X/Y 坐标的本质
网格化分区:指纹图像被划分为若干个小块（如 16×16 像素的网格），
每个块对应 .qm 文件中的一个数值（质量分）。
"""