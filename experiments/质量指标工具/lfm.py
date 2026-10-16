"""
这个文件（.lfm 文件）是一个二维矩阵，数值为 0 或 1，
可能表示某种局部特征标记（Local Feature Map）。
`.lfm` 文件中的标记点（`1`）可能反映了指纹细节特征、图像目标区域或逻辑状态编码，
具体意义需结合生成背景和原始数据验证。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 关键修复！
plt.rcParams['axes.unicode_minus'] = False


def read_lfm_file(file_path):
    """
    读取.lfm文件并解析为二维数组
    参数:
        file_path: 文件路径（字符串）
    返回:
        lfm_data: 二维numpy数组（0和1组成的矩阵）
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行

    # 过滤空行并转换数据
    data = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # 跳过空行
            row = list(map(int, stripped_line.split()))  # 分割字符串并转为整数
            data.append(row)

    return np.array(data)


def analyze_lfm(lfm_data):
    """
    分析.lfm文件数据
    参数:
        lfm_data: 二维numpy数组
    """
    # ==================== 1. 基础统计 ====================
    print(f"数据形状: {lfm_data.shape} (行×列)")
    print(f"标记点总数: {np.sum(lfm_data)}")
    print(f"标记点占比: {np.mean(lfm_data):.2%}")

    # ==================== 2. 可视化 ====================
    plt.figure(figsize=(10, 6))

    # 热力图显示标记分布
    plt.imshow(lfm_data, cmap='binary', interpolation='nearest')
    plt.colorbar(label='标记值 (0/1)')
    plt.title("LFM 文件标记分布热力图")
    plt.xlabel("X坐标（列）")
    plt.ylabel("Y坐标（行）")

    # 标记所有值为1的点
    y_coords, x_coords = np.where(lfm_data == 1)
    plt.scatter(x_coords, y_coords, color='red', s=10, label='标记点(1)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ==================== 3. 高级分析 ====================
    # 计算标记点的聚集程度（最近邻距离）
    from scipy.spatial import KDTree
    if len(x_coords) > 1:
        points = np.column_stack((x_coords, y_coords))
        kdtree = KDTree(points)
        distances, _ = kdtree.query(points, k=2)  # 每个点与最近邻的距离
        avg_min_distance = np.mean(distances[:, 1])  # 忽略自身
        print(f"标记点平均最近邻距离: {avg_min_distance:.2f} 单位")
    else:
        print("警告：标记点过少，无法计算聚集度")

    # 检测连续标记区域（连通域分析）
    from scipy.ndimage import label
    labeled_array, num_features = label(lfm_data)
    print(f"连通区域数量: {num_features}")

    # 可视化连通区域
    if num_features > 0:
        plt.figure(figsize=(10, 6))
        plt.imshow(labeled_array, cmap='tab20b')
        plt.colorbar(label='连通区域ID')
        plt.title("连通区域分析")
        plt.show()


# 示例使用
if __name__ == "__main__":
    file_path = "26_3_features.lfm"  # 替换为实际文件路径
    lfm_data = read_lfm_file(file_path)
    analyze_lfm(lfm_data)
# 生成热力图（带标记点）
plt.figure(figsize=(10, 6))
plt.imshow(lfm_data, cmap='binary')
plt.colorbar(label='0/1标记')
plt.scatter(*np.where(lfm_data==1)[::-1], c='red', s=10, label='标记点')
plt.title("优化后的标记分布")
plt.show()
