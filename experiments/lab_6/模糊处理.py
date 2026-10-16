import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def process_image(img_path, output_dir):
    """
    处理单张图像：应用随机几何变换和模糊效果，生成模糊版本图像
    :param img_path: 输入图像路径
    :param output_dir: 输出目录路径
    """
    try:
        # 1. 读取图像并转换为灰度模式
        img = Image.open(img_path)  # 打开图像文件，支持多种格式（由Pillow库决定）
        img = img.convert('L')  # 转换为灰度模式（'L'模式，像素值范围0-255）
        original_size = img.size  # 保存原始图像尺寸（宽度, 高度）

        # --------------------------- 几何变换部分 ---------------------------
        # 2. 随机旋转变换（-10°到10°之间）
        angle = random.uniform(-10, 10)  # 生成随机角度（浮点数，包含正负值）
        # rotate方法参数说明：
        # - angle: 旋转角度（正数为逆时针，负数为顺时针）
        # - expand=True: 自动扩展图像尺寸以容纳完整旋转后的图像
        # - fillcolor='white': 旋转后空白区域用白色填充（灰度值255）
        rotated = img.rotate(angle, expand=True, fillcolor='white')
        # 调整旋转后的图像尺寸回原始尺寸（可能导致边缘拉伸，需谨慎）
        rotated = rotated.resize(original_size)  # resize((width, height))

        # 3. 随机平移变换（水平/垂直偏移-20到20像素）
        x_offset = random.randint(-20, 20)  # 水平偏移量（像素值，负数向左，正数向右）
        y_offset = random.randint(-20, 20)  # 垂直偏移量（像素值，负数向上，正数向下）
        # transform方法实现仿射变换：
        # - Image.AFFINE表示仿射变换类型
        # - 变换矩阵参数：(a, b, c, d, e, f) 对应变换公式：x' = a*x + b*y + c; y' = d*x + e*y + f
        # 此处实现单纯平移，故a=1, b=0, d=0, e=1，仅c=x_offset, f=y_offset
        translated = rotated.transform(
            original_size,  # 输出图像尺寸保持原始尺寸
            Image.AFFINE,  # 变换类型为仿射变换
            (1, 0, x_offset, 0, 1, y_offset),  # 仿射变换矩阵
            fillcolor='white'  # 偏移后空白区域用白色填充
        )

        # --------------------------- 模糊处理部分 ---------------------------
        # 4. 随机选择模糊类型（高斯模糊/方框模糊/运动模糊）
        blur_type = random.choice(['gaussian', 'box', 'motion'])  # 从列表中随机选取一种类型

        if blur_type == 'gaussian':
            # 高斯模糊：半径控制模糊程度（半径越大越模糊）
            radius = random.uniform(0.5, 2.5)  # 生成0.5到2.5之间的随机半径（浮点数）
            # 使用Pillow内置的高斯模糊滤镜
            processed = translated.filter(ImageFilter.GaussianBlur(radius))

        elif blur_type == 'box':
            # 方框模糊：radius表示模糊核大小（必须为正整数，代表相邻像素的范围）
            radius = random.randint(1, 3)  # 生成1到3之间的随机整数（对应1x1到3x3的模糊核）
            processed = translated.filter(ImageFilter.BoxBlur(radius))

        else:  # motion blur 运动模糊（需手动构造卷积核）
            kernel_size = 9  # 固定使用9x9大小的卷积核（奇数尺寸便于中心对齐）
            kernel = np.zeros((kernel_size, kernel_size))  # 创建全零的核矩阵

            # 随机选择水平或垂直方向的运动模糊
            if random.random() < 0.5:
                # 水平模糊：核的中间行全为1/核大小（模拟水平方向运动）
                kernel[kernel_size // 2, :] = 1.0 / kernel_size  # kernel_size//2是中间行索引
            else:
                # 垂直模糊：核的中间列全为1/核大小（模拟垂直方向运动）
                kernel[:, kernel_size // 2] = 1.0 / kernel_size

            # 将numpy数组转换为Pillow所需的一维列表格式（按行优先展开）
            kernel = kernel.flatten().tolist()  # 例如9x9矩阵转为81个元素的列表

            # 尝试创建自定义卷积核滤镜（可能因参数错误失败，需异常处理）
            try:
                kernel_filter = ImageFilter.Kernel(
                    size=(kernel_size, kernel_size),  # 核尺寸（宽, 高）
                    kernel=kernel,  # 核权重值（一维列表）
                    scale=1.0,  # 缩放因子（通常设为1.0，即不缩放）
                    offset=0.0  # 偏移量（像素值偏移，通常为0）
                )
                processed = translated.filter(kernel_filter)
            except ValueError as e:
                # 若核创建失败（如尺寸不匹配、权重和不为1等），回退到高斯模糊
                print(f"Motion blur kernel error: {e}, fallback to Gaussian blur")
                processed = translated.filter(ImageFilter.GaussianBlur(1.5))

        # --------------------------- 保存结果部分 ---------------------------
        # 5. 保存处理后的图像
        base_name = os.path.basename(img_path)  # 从路径中提取文件名（如"image.jpg"）
        output_path = os.path.join(output_dir, base_name)  # 拼接输出路径

        # 确保图像模式为'L'（灰度），避免因模式错误导致保存异常
        if processed.mode != 'L':
            processed = processed.convert('L')  # 强制转换为灰度模式

        processed.save(output_path)  # 保存图像（格式由文件名后缀决定，如.jpg/.png）
        print(f"Successfully processed: {base_name}")

    except Exception as e:
        # 捕获处理过程中的所有异常并打印详细信息
        print(f"Error processing {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈信息（调试用）


def main():
    """主函数：批量处理输入目录中的图像，生成模糊版本到输出目录"""
    # 配置输入输出路径（需根据实际情况修改）
    input_dir = r"C:\Users\86158\Desktop\D\D_data\Example Data\gallery"  # 原始清晰图像目录
    output_dir = r"data_x"  # 处理后的模糊图像保存目录

    # 确保输出目录存在，不存在则创建（exist_ok=True表示若存在不报错）
    os.makedirs(output_dir, exist_ok=True)

    # 筛选有效图像文件（支持常见图片格式，忽略大小写）
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # 元组用于多条件判断
    image_files = [
        f for f in os.listdir(input_dir)  # 遍历输入目录所有文件
        if f.lower().endswith(valid_ext)  # 检查文件后缀（转小写后匹配）
    ]

    print(f"Found {len(image_files)} images to process")  # 打印待处理文件数量

    # 逐个处理图像文件
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)  # 拼接完整输入路径
        process_image(img_path, output_dir)  # 调用处理函数

    print("Processing completed!")  # 处理完成提示


if __name__ == "__main__":
    main()  # 当脚本直接运行时执行主函数