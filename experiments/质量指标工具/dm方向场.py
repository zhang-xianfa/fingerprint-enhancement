import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 关键修复！
plt.rcParams['axes.unicode_minus'] = False
def read_dm_file(file_path):
    """读取文本格式的.dm文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析数据为二维数组
    data = []
    for line in lines:
        row = [int(x) if x != '-1' else -1 for x in line.strip().split()]
        if row:  # 跳过空行
            data.append(row)

    return np.array(data)


# 使用示例
dm_file = "26_3_features.dm"  # 替换为你的文件路径
dm_data = read_dm_file(dm_file)
print("数据形状:", dm_data.shape)
print("示例数据:\n", dm_data[:5, :5])  # 查看前5行5列
import numpy as np
import matplotlib.pyplot as plt


def read_orientation_file(file_path):
    """
    读取方向场文件，返回矩阵
    参数:
        file_path: 文件路径
    返回:
        orientation: 二维NumPy数组（无效值为-1，有效值为0-15）
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    orientation = np.array([[int(num) for num in line.strip().split()] for line in lines if line.strip()])
    return orientation


def plot_orientation_field(orientation):
    """
    可视化方向场
    参数:
        orientation: 方向场矩阵（值范围-1~15）
    """
    # 将量化值转换为角度（弧度）
    valid_mask = (orientation >= 0)  # 忽略-1（无效区域）
    angles = np.radians(orientation[valid_mask] * 11.25)  # 每个值代表11.25°的区间

    # 生成网格坐标
    rows, cols = np.where(valid_mask)
    x = cols
    y = rows

    # 绘制方向场（箭头表示方向）
    plt.figure(figsize=(12, 8))
    plt.quiver(x, y, np.cos(angles), np.sin(angles),
               scale=20, width=0.002, headwidth=3, color='blue')

    # 标记无效区域（-1）
    invalid_mask = (orientation == -1)
    invalid_rows, invalid_cols = np.where(invalid_mask)
    plt.scatter(invalid_cols, invalid_rows, color='red', s=5, label='无效区域')

    plt.title("指纹方向场可视化（箭头方向=脊线走向）")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.legend()
    plt.gca().invert_yaxis()  # 图像坐标系Y轴向下
    plt.show()


# 示例使用
if __name__ == "__main__":
    file_path = "26_3_features.dm"  # 替换为实际文件路径
    orientation = read_orientation_file(file_path)
    print("方向场矩阵形状:", orientation.shape)

    # 统计有效方向区域占比
    valid_ratio = np.mean(orientation >= 0)
    print(f"有效方向区域占比: {valid_ratio:.2%}")

    plot_orientation_field(orientation)
