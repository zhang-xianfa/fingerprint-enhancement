import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# 修改后的生成器（适配灰度图像）
class Generator(nn.Module):
    """
    修改说明：
    1. 输入输出通道从3改为1
    2. 增加网络深度提升生成质量
    3. 调整激活函数配置
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 编码器部分
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 输入通道改为1
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 新增深度层
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 解码器部分
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 输出通道改为1
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


def load_generator(model_path, device):
    """加载模型时添加strict=False兼容可能的参数变化"""
    generator = Generator().to(device)
    state_dict = torch.load(model_path, map_location=device)

    # 处理可能的参数名称变化
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):  # 处理多GPU训练保存的模型
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    generator.load_state_dict(new_state_dict, strict=False)
    generator.eval()
    return generator


def preprocess_image(image_path, transform):
    """改进的预处理函数"""
    # 强制转换为灰度图并应用转换
    image = Image.open(image_path).convert('L')  # 确保灰度模式
    print(f"original image mode:{image.mode}")
    image = transform(image)
    print(f"tensor shape:{image.shape}")
    return image.unsqueeze(0)  # 添加batch维度 (1,1,H,W)


def postprocess_image(tensor):
    """适配单通道的后处理"""
    tensor = tensor.squeeze(0)  # 移除batch维度 (1,H,W)
    tensor = (tensor + 1) / 2  # 反归一化到[0,1]
    tensor = tensor.clamp(0, 1)  # 确保数值范围正确
    return transforms.ToPILImage()(tensor.cpu())


def infer_images(generator, latent_dir, output_dir, transform, device):
    """改进的推理函数"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有PNG和JPG文件
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_names = [f for f in os.listdir(latent_dir)
                   if f.lower().endswith(valid_exts)]

    for image_name in image_names:
        latent_path = os.path.join(latent_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        try:
            # 预处理
            latent_tensor = preprocess_image(latent_path, transform).to(device)

            # 推理
            with torch.no_grad():
                enhanced = generator(latent_tensor)

            # 保存结果
            postprocess_image(enhanced).save(output_path)
            print(f"Success: {image_name}")

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 适配灰度图的预处理流程
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 单通道归一化
    ])

    # 模型路径配置（根据实际路径修改）
    generator = load_generator('generator_final.pth', device)

    # 路径配置
    latent_dir = 'latent'  # 输入目录
    output_dir = 'enhanced_output'  # 输出目录

    # 执行增强
    infer_images(generator, latent_dir, output_dir, transform, device)


if __name__ == '__main__':
    main()