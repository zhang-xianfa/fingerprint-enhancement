from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------- 配置参数 ----------------------
folder_path = "gallery"  # 待分析的文件夹路径（改为你的实际路径）
output_plot_path = "gallery_size_statistics.png"  # 统计图表保存路径
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')  # 支持的图像格式
abnormal_threshold = 0.2  # 定义异常尺寸阈值（与最常见尺寸差异百分比）

# ---------------------- 1. 遍历图像并收集尺寸数据 ----------------------
size_data = []  # 存储所有有效图像的尺寸
size_map = {}  # 映射文件名到尺寸（用于后续关联异常文件）
invalid_files = []  # 存储读取失败的文件

for filename in os.listdir(folder_path):
    # 过滤非图像文件
    if not filename.lower().endswith(valid_extensions):
        continue

    img_path = os.path.join(folder_path, filename)
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            size_data.append((width, height))
            size_map[filename] = (width, height)  # 建立文件名到尺寸的映射
    except Exception as e:
        invalid_files.append((filename, str(e)))


# ---------------------- 2. 数据统计分析 ----------------------
def analyze_size_data(size_data):
    """分析尺寸数据并返回统计结果"""
    if not size_data:
        return None

    # 提取所有宽度和高度
    widths = [w for w, h in size_data]
    heights = [h for w, h in size_data]

    # 统计唯一尺寸及其出现次数
    size_counts = defaultdict(int)
    for w, h in size_data:
        size_counts[(w, h)] += 1

    # 按出现次数排序
    sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[1])

    return {
        "total_images": len(size_data),
        "unique_sizes": len(size_counts),
        "most_common_size": sorted_sizes[0] if sorted_sizes else None,
        "width_stats": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths)
        },
        "height_stats": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights)
        },
        "size_distribution": sorted_sizes
    }


# ---------------------- 3. 绘制尺寸分布直方图 ----------------------
def plot_size_distribution(size_data, output_path, common_size=None, threshold=0.2):
    """绘制宽度和高度的分布直方图，并标记异常区域"""
    if not size_data:
        print("警告：无有效图像数据，无法绘图。")
        return

    widths = [w for w, h in size_data]
    heights = [h for w, h in size_data]

    plt.figure(figsize=(12, 8))

    # 宽度直方图
    ax1 = plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(widths, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Width Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")

    # 标记最常见尺寸和异常区域
    if common_size:
        common_w, common_h = common_size
        plt.axvline(common_w, color='blue', linestyle='--', label=f"Most common: {common_w}")
        plt.axvline(common_w * (1 - threshold), color='red', linestyle=':', label=f"±{threshold * 100}% threshold")
        plt.axvline(common_w * (1 + threshold), color='red', linestyle=':')
        plt.legend()

    # 高度直方图
    ax2 = plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Height Distribution")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Frequency")

    # 标记最常见尺寸和异常区域
    if common_size:
        plt.axvline(common_h, color='blue', linestyle='--')
        plt.axvline(common_h * (1 - threshold), color='red', linestyle=':')
        plt.axvline(common_h * (1 + threshold), color='red', linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"统计图表已保存至：{output_path}")
    plt.close()


# ---------------------- 4. 执行分析并输出结果 ----------------------
if __name__ == "__main__":
    # 分析数据
    stats = analyze_size_data(size_data)

    # 打印统计摘要
    print(f"\n===== {folder_path} 图像尺寸统计分析 =====")
    print(f"总图像数：{stats['total_images']}")
    print(f"唯一尺寸数量：{stats['unique_sizes']}")

    if stats['most_common_size']:
        common_size = stats['most_common_size'][0]
        print(f"最常见尺寸：{common_size}（出现 {stats['most_common_size'][1]} 次）")

        # 检测异常尺寸
        abnormal_files = []
        for filename, (w, h) in size_map.items():
            width_diff = abs(w - common_size[0]) / common_size[0] if common_size[0] != 0 else 0
            height_diff = abs(h - common_size[1]) / common_size[1] if common_size[1] != 0 else 0

            if width_diff > abnormal_threshold or height_diff > abnormal_threshold:
                abnormal_files.append((filename, w, h, width_diff, height_diff))

        # 输出异常尺寸文件详情（按差异程度排序）
        print(f"\n===== 异常尺寸图像列表（差异>{abnormal_threshold * 100}%） =====")
        if abnormal_files:
            abnormal_files.sort(key=lambda x: max(x[3], x[4]), reverse=True)
            for fname, w, h, w_diff, h_diff in abnormal_files:
                print(f"文件名：{fname}, 尺寸：{w}x{h}, 差异：宽{w_diff:.1%} / 高{h_diff:.1%}")
        else:
            print("未检测到异常尺寸图像。")

        # 绘制带异常标记的直方图
        plot_size_distribution(size_data, output_plot_path, common_size, abnormal_threshold)
    else:
        print("警告：未找到常见尺寸（数据集可能为空或尺寸完全随机）。")
        plot_size_distribution(size_data, output_plot_path)

    # 打印读取失败的文件
    if invalid_files:
        print(f"\n警告：以下 {len(invalid_files)} 个文件读取失败：")
        for fname, err in invalid_files:
            print(f"- {fname}: {err}")