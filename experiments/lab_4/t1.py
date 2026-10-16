import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_periodic_noise(image):
    # 傅里叶变换
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # 设计高通滤波器（示例：去除中心低频噪声）
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0  # 过滤中心区域

    # 应用掩模并逆变换
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    return img_filtered.astype(np.uint8)


def enhance_contrast(image):
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(image)

    # 伽马校正（γ=0.7）
    gamma = 0.7
    img_gamma = np.power(img_clahe / 255.0, gamma) * 255.0

    return img_gamma.astype(np.uint8)


def enhance_ridges(image):
    # Gabor滤波器（示例：4个方向）
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    # 应用Gabor滤波并取最大值
    img_gabor = np.zeros_like(image, dtype=np.float32)
    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        img_gabor = np.maximum(img_gabor, filtered)

    # 归一化到[0, 255]
    img_gabor = cv2.normalize(img_gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 形态学操作（闭运算修复断裂）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_gabor, cv2.MORPH_CLOSE, kernel)

    return img_morph


def resize_and_normalize(image, target_size=(256, 256)):
    # 缩放图像
    img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 归一化
    img_normalized = img_resized / 255.0

    return img_normalized.astype(np.float32)


def preprocess_latent(image):
    # 1. 频域去噪
    img_filtered = remove_periodic_noise(image)

    # 2. 对比度增强
    img_contrast = enhance_contrast(img_filtered)

    # 3. 脊线增强
    img_ridges = enhance_ridges(img_contrast)

    # 4. 尺寸标准化与归一化
    img_final = resize_and_normalize(img_ridges)

    return img_final


# 新增：批量处理主函数
def batch_preprocess_latent_images(input_dir, output_dir, target_size=(256, 256)):
    """
    批量预处理latent文件夹中的所有图像

    参数:
        input_dir: 输入latent图像文件夹路径
        output_dir: 输出预处理图像文件夹路径
        target_size: 目标图像尺寸
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"发现 {len(image_files)} 张图像需要预处理...")

    # 遍历所有图像进行预处理
    for filename in tqdm(image_files, desc="预处理进度"):
        try:
            # 读取图像
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 预处理
            processed_image = preprocess_latent(image)

            # 保存预处理后的图像（保存为float32类型的npy文件）
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")
            np.save(output_path, processed_image)

        except Exception as e:
            print(f"处理图像 {filename} 时出错: {str(e)}")

    print(f"预处理完成! 所有图像已保存到 {output_dir}")
    return output_dir


# 新增：加载预处理图像的函数
def load_preprocessed_image(file_path):
    """加载预处理后的图像（从npy文件）"""
    return np.load(file_path)


# 主程序入口
if __name__ == "__main__":
    # 配置参数
    LATENT_DIR = "gallery"  # 原始latent图像文件夹
    PROCESSED_DIR = "processed_gallery"  # 预处理后图像保存文件夹
    TARGET_SIZE = (256, 256)  # 目标图像尺寸

    # 批量预处理
    processed_dir = batch_preprocess_latent_images(
        input_dir=LATENT_DIR,
        output_dir=PROCESSED_DIR,
        target_size=TARGET_SIZE
    )

    # 示例：加载一张预处理后的图像
    sample_file = os.listdir(processed_dir)[0]
    sample_path = os.path.join(processed_dir, sample_file)
    sample_image = load_preprocessed_image(sample_path)

    print(f"示例图像形状: {sample_image.shape}")
    print(f"示例图像数据类型: {sample_image.dtype}")
    print(f"示例图像值范围: [{sample_image.min()}, {sample_image.max()}]")