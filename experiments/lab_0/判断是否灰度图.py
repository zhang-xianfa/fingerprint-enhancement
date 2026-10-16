from PIL import Image

from PIL import Image
import numpy as np


def is_grayscale_pillow(image_path):
    """
    使用Pillow库判断图片是否为标准灰度格式（模式为'L'）

    参数:
        image_path (str): 图片文件路径，需确保路径存在且为有效图像文件

    返回值:
        bool:
            - True  表示图片是标准灰度格式（模式为'L'）
            - False 表示图片非灰度格式，或读取失败

    涉及的Python语法:
        1. 函数定义（def语句）
        2. 异常处理（try-except块）
        3. 上下文管理器（Image.open隐式使用）
        4. 对象属性访问（img.mode）
        5. 布尔比较（==运算符）
        6. 格式化字符串（f-string）
        7. 返回布尔值（True/False）

    图像处理知识:
        1. Pillow库的Image.open()方法
        2. 图像模式（mode）概念：
           - 'L'：8位灰度（0-255）
           - 'RGB'：24位真彩色
           - 'RGBA'：带透明通道的彩色
        3. 标准灰度图的定义：每个像素只有亮度值，无色彩信息

    注意事项:
        1. 仅检测图像格式是否为'L'，不检查像素内容
        2. 某些实际为灰度的图片可能以RGB模式存储（三通道数值相同）
        3. 需要先安装Pillow库：pip install Pillow
    """
    try:
        # 打开图像文件（自动关闭文件句柄）
        # 可能触发FileNotFoundError/UnidentifiedImageError等异常
        img = Image.open(image_path)

        # 严格判断图像模式
        # 注意：不检查像素内容，仅验证存储格式
        return img.mode == 'L'  # 返回布尔比较结果

    except Exception as e:
        # 捕获所有异常，包括：
        # - 文件不存在（FileNotFoundError）
        # - 非图像文件（PIL.UnidentifiedImageError）
        # - 权限错误（PermissionError）等
        print(f"Error: {e}")  # 打印具体错误信息
        return False  # 统一返回False表示判断失败


def is_grayscale(image_path):
    """
    严格判断图片是否为灰度图（兼顾格式和实际像素）

    参数:
        image_path: 图片路径

    返回:
        tuple: (是否灰度图, 失败原因/None)
    """
    try:
        img = Image.open(image_path)

        # 检查1：格式层面（模式是否为'L'）
        if img.mode == 'L':
            return (True, None)

        # 检查2：像素层面（所有像素RGB是否相同）
        if img.mode in ('RGB', 'RGBA'):
            arr = np.array(img)
            if len(arr.shape) == 3:  # 确保是多通道图片
                # 比较所有像素的R/G/B通道是否一致
                if (arr[:, :, 0] == arr[:, :, 1]).all() and (arr[:, :, 1] == arr[:, :, 2]).all():
                    return (True, "视觉灰度（实际为RGB模式）")

        return (False, f"模式为{img.mode}且像素非灰度")

    except Exception as e:
        return (False, f"读取失败: {str(e)}")

# 示例
image_path = "26_2.png"
if is_grayscale_pillow(image_path):
    print("该图片是灰度图")
else:
    print("该图片不是灰度图")
