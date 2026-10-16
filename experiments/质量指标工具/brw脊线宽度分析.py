import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path


def auto_detect_brw(file_path, max_attempts=5):
    """
    自动检测并读取.brw文件
    返回：
        data (np.ndarray): 解析出的矩阵数据
        meta (dict): 包含文件格式信息
    """
    file_size = os.path.getsize(file_path)
    meta = {
        'file_size': file_size,
        'detected_dtype': None,
        'detected_shape': None,
        'header_bytes': 0
    }

    # 尝试常见数据类型 (按优先级排序)
    dtypes = [
        ('float32', np.float32, 4),
        ('uint8', np.uint8, 1),
        ('uint16', np.uint16, 2),
        ('int16', np.int16, 2),
        ('float64', np.float64, 8)
    ]

    for attempt in range(max_attempts):
        for dtype_name, dtype, byte_size in dtypes:
            try:
                # 尝试不同可能的头部偏移量
                header_bytes = attempt * 128  # 常见头大小步进
                remaining_bytes = file_size - header_bytes

                if remaining_bytes <= 0:
                    continue

                num_elements = remaining_bytes // byte_size
                if num_elements < 4:  # 至少2x2矩阵
                    continue

                # 计算最可能的矩阵形状 (优先方形)
                dim = math.isqrt(num_elements)
                shape = (dim, dim)

                with open(file_path, 'rb') as f:
                    if header_bytes > 0:
                        f.read(header_bytes)
                    data = np.fromfile(f, dtype=dtype)

                # 截断多余数据
                expected_size = shape[0] * shape[1]
                if len(data) > expected_size:
                    data = data[:expected_size]

                data = data.reshape(shape)

                # 验证数据合理性 (排除全零或异常值)
                if np.all(data == 0) or np.max(data) > 1e6:
                    continue

                meta.update({
                    'detected_dtype': dtype_name,
                    'detected_shape': shape,
                    'header_bytes': header_bytes,
                    'used_elements': len(data)
                })
                return data, meta

            except Exception as e:
                continue

    raise ValueError(f"无法自动识别 {file_path} 的格式，请手动指定参数")


def analyze_brw(file_path):
    """完整分析.brw文件"""
    print(f"分析文件: {Path(file_path).name}")

    # 1. 自动检测读取
    data, meta = auto_detect_brw(file_path)
    print("\n=== 文件格式信息 ===")
    for k, v in meta.items():
        print(f"{k:>15}: {v}")

    # 2. 基本统计
    print("\n=== 数据统计 ===")
    print(f"{'Shape':>15}: {data.shape}")
    print(f"{'Min':>15}: {np.min(data):.2f}")
    print(f"{'Max':>15}: {np.max(data):.2f}")
    print(f"{'Mean':>15}: {np.mean(data):.2f}")
    print(f"{'Std':>15}: {np.std(data):.2f}")

    # 3. 可视化
    plt.figure(figsize=(15, 5))

    # 原始数据
    plt.subplot(131)
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Raw Data')

    # 直方图
    plt.subplot(132)
    plt.hist(data.flatten(), bins=50, log=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency (log)')
    plt.title('Value Distribution')

    # 二值化预览 (自动阈值)
    plt.subplot(133)
    threshold = np.median(data)
    binary = (data > threshold).astype(np.uint8)
    plt.imshow(binary, cmap='gray')
    plt.title(f'Binary (Threshold={threshold:.2f})')

    plt.tight_layout()
    plt.show()

    return data


# 使用示例
if __name__ == "__main__":
    brw_file = "26_3_features.brw"  # 替换为你的文件路径
    try:
        brw_data = analyze_brw(brw_file)

        # 示例：提取特征区域
        threshold = np.quantile(brw_data, 0.75)
        features = brw_data > threshold
        print(f"\n特征区域占比: {np.mean(features):.2%}")

    except Exception as e:
        print(f"分析失败: {str(e)}")
        print("建议：")
        print("1. 检查文件路径是否正确")
        print("2. 确认文件未被占用")
        print("3. 联系数据提供方确认格式")
