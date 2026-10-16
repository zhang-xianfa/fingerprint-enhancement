import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.nn.utils import spectral_norm

# 配置参数
# 图像高度和宽度，设置为512x512像素
IMG_HEIGHT = 512
IMG_WIDTH = 512
# 图像通道数，1表示灰度图像
IMG_CHANNELS = 1
# 批次大小，每次训练处理的样本数量
BATCH_SIZE = 4
# 训练轮数，整个数据集被训练的次数
EPOCHS = 100
# L1损失的权重系数，控制生成图像与目标图像的相似程度
LAMBDA = 50
# 设备选择，优先使用GPU，否则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 生成器定义
# 基于U-Net架构的生成器，用于将模糊指纹图像转换为清晰图像
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器 (下采样)
        # 每个DownBlock将图像尺寸缩小一半，同时增加通道数
        self.enc1 = DownBlock(IMG_CHANNELS, 64, batchnorm=False)  # 512->256
        self.enc2 = DownBlock(64, 128)  # 256->128
        self.enc3 = DownBlock(128, 256)  # 128->64
        self.enc4 = DownBlock(256, 512)  # 64->32
        self.enc5 = DownBlock(512, 512)  # 32->16
        self.enc6 = DownBlock(512, 512)  # 16->8
        self.enc7 = DownBlock(512, 512)  # 8->4
        self.enc8 = DownBlock(512, 512)  # 4->2

        # 解码器 (上采样)
        # 每个UpBlock将图像尺寸扩大一倍，同时减少通道数
        # 部分块使用dropout防止过拟合
        self.dec1 = UpBlock(512, 512, dropout=True)  # 2->4
        self.dec2 = UpBlock(1024, 512, dropout=True)  # 4->8
        self.dec3 = UpBlock(1024, 512, dropout=True)  # 8->16
        self.dec4 = UpBlock(1024, 512)  # 16->32
        self.dec5 = UpBlock(1024, 256)  # 32->64
        self.dec6 = UpBlock(512, 128)  # 64->128
        self.dec7 = UpBlock(256, 64)  # 128->256

        # 最后一层
        # 使用反卷积和Tanh激活函数生成最终图像
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, IMG_CHANNELS, 4, 2, 1),
            nn.Tanh()  # why
        )

    def forward(self, x):
        # 编码器
        # 保存每一层的输出用于跳跃连接
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # 解码器 (带跳跃连接)
        # 将编码器对应层的特征图与解码器的特征图拼接
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], 1)  # 沿着通道维度拼接

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], 1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], 1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], 1)

        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], 1)

        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], 1)

        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], 1)

        return self.final(d7)


# 下采样块
# 用于编码器，通过卷积和下采样减少图像尺寸
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(DownBlock, self).__init__()
        layers = [
            # 卷积层，步长为2实现下采样
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        ]
        if batchnorm:
            # 批量归一化层，加速训练并提高稳定性
            layers.append(nn.BatchNorm2d(out_channels))
        # LeakyReLU激活函数，负值部分保留小梯度
        layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 上采样块
# 用于解码器，通过反卷积和上采样增加图像尺寸
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpBlock, self).__init__()
        layers = [
            # 反卷积层，步长为2实现上采样
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            # 批量归一化层
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数
            nn.ReLU()
        ]
        if dropout:
            # Dropout层，随机丢弃部分神经元防止过拟合
            layers.insert(2, nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 多尺度判别器
# 同时在不同分辨率下判断图像真伪
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        # 完整分辨率判别器 (512x512)
        self.discriminator_full = self._make_discriminator()

        # 1/2分辨率判别器 (256x256)
        self.discriminator_half = self._make_discriminator()

        # 1/4分辨率判别器 (128x128)
        self.discriminator_quarter = self._make_discriminator()

        # 平均池化层，用于下采样
        self.downsample = nn.AvgPool2d(2, stride=2, padding=0)

    def _make_discriminator(self):
        return nn.Sequential(
            # 输入是 (IMG_CHANNELS*2) x H x W，拼接的输入和目标/生成图像
            spectral_norm(nn.Conv2d(IMG_CHANNELS * 2, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            # ZeroPad2d用于调整图像尺寸
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)),
            nn.LeakyReLU(0.2),

            nn.ZeroPad2d((1, 0, 1, 0)),
            # 输出单通道预测图，表示每个区域的真伪概率
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, img_A, img_B):
        # 完整分辨率
        input_full = torch.cat((img_A, img_B), 1)
        out_full = self.discriminator_full(input_full)

        # 1/2分辨率
        input_half = self.downsample(input_full)
        out_half = self.discriminator_half(input_half)

        # 1/4分辨率
        input_quarter = self.downsample(input_half)
        out_quarter = self.discriminator_quarter(input_quarter)

        # 返回三个尺度的判别结果
        return [out_full, out_half, out_quarter]


# 损失函数
# 计算生成器的损失
def generator_loss(disc_outputs, gen_output, target):
    total_loss = 0
    gan_loss = 0
    # L1损失，计算生成图像与目标图像的逐像素绝对差异
    l1_loss = nn.L1Loss()(gen_output, target)

    # 多尺度判别损失
    # 每个尺度的判别器输出都计算对抗损失
    for disc_out in disc_outputs:
        current_loss = nn.BCEWithLogitsLoss()(disc_out, torch.ones_like(disc_out))
        total_loss += current_loss
        gan_loss += current_loss

    # 总损失 = 对抗损失 + LAMBDA * L1损失
    total_loss += LAMBDA * l1_loss
    return total_loss, gan_loss, l1_loss


# 计算判别器的损失
def discriminator_loss(disc_real_outputs, disc_gen_outputs):
    total_loss = 0

    # 对每个尺度的判别器输出，分别计算真实图像和生成图像的损失
    for d_real, d_gen in zip(disc_real_outputs, disc_gen_outputs):
        # 真实图像应该被判别为1
        real_loss = nn.BCEWithLogitsLoss()(d_real, torch.ones_like(d_real))
        # 生成图像应该被判别为0
        gen_loss = nn.BCEWithLogitsLoss()(d_gen, torch.zeros_like(d_gen))
        total_loss += real_loss + gen_loss

    return total_loss


# 数据集类
# 加载配对的模糊指纹和清晰指纹图像
class PairedFingerprintDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        # 确保两个目录中的文件一一对应
        self.blur_files = sorted(os.listdir(blur_dir))
        self.sharp_files = sorted(os.listdir(sharp_dir))

        # 验证文件名是否匹配
        assert len(self.blur_files) == len(self.sharp_files), "两个目录中的文件数量不匹配"
        for b, s in zip(self.blur_files, self.sharp_files):
            assert b == s, f"文件名不匹配: {b} != {s}"

        # 构建完整文件路径
        self.blur_paths = [os.path.join(blur_dir, f) for f in self.blur_files]
        self.sharp_paths = [os.path.join(sharp_dir, f) for f in self.sharp_files]

        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
        ])

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        # 读取图像，使用OpenCV的灰度模式
        blur_img = cv2.imread(self.blur_paths[idx], cv2.IMREAD_GRAYSCALE)
        sharp_img = cv2.imread(self.sharp_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 调整图像大小为指定尺寸
        blur_img = cv2.resize(blur_img, (IMG_WIDTH, IMG_HEIGHT))
        sharp_img = cv2.resize(sharp_img, (IMG_WIDTH, IMG_HEIGHT))

        # 应用预处理转换
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)

        return blur_tensor, sharp_tensor


# 生成示例图像
# 用于可视化训练过程中的生成效果
def generate_images(model, test_input, target, epoch, save_dir):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        prediction = model(test_input)
    model.train()  # 恢复模型为训练模式

    # 将张量转换为numpy数组并移到CPU上
    test_input = test_input.cpu().numpy()
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # 创建图像显示
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # 显示输入图像、目标图像和生成图像
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # 反归一化：将[-1,1]范围的像素值恢复到[0,1]范围
        plt.imshow(display_list[i].squeeze() * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch + 1}.png'))
    plt.close()


# 单步训练函数
# 执行一个训练步骤，包括判别器和生成器的前向和反向传播
def train_step(blur_imgs, sharp_imgs, generator, discriminator, g_optimizer, d_optimizer):
    blur_imgs = blur_imgs.to(DEVICE)  # 将数据移到指定设备
    sharp_imgs = sharp_imgs.to(DEVICE)

    # 训练判别器
    d_optimizer.zero_grad()  # 清零判别器的梯度

    # 真实图像
    d_real_outputs = discriminator(blur_imgs, sharp_imgs)

    # 生成图像
    gen_output = generator(blur_imgs)
    d_gen_outputs = discriminator(blur_imgs, gen_output.detach())
    # 使用detach()避免计算生成器的梯度

    d_loss = discriminator_loss(d_real_outputs, d_gen_outputs)
    d_loss.backward()  # 反向传播
    d_optimizer.step()  # 更新判别器参数

    # 训练生成器
    g_optimizer.zero_grad()  # 清零生成器的梯度

    gen_output = generator(blur_imgs)
    d_gen_outputs = discriminator(blur_imgs, gen_output)

    g_total_loss, g_gan_loss, g_l1_loss = generator_loss(
        d_gen_outputs, gen_output, sharp_imgs)
    g_total_loss.backward()  # 反向传播
    g_optimizer.step()  # 更新生成器参数

    return d_loss.item(), g_total_loss.item(), g_gan_loss.item(), g_l1_loss.item()


# 主训练函数
def train(blur_dir, sharp_dir, epochs=EPOCHS):
    # 创建保存目录
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = f'./training_checkpoints_{current_time}'
    sample_dir = f'./samples_{current_time}'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 数据集和数据加载器
    dataset = PairedFingerprintDataset(blur_dir, sharp_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 创建模型
    generator = Generator().to(DEVICE)
    discriminator = MultiScaleDiscriminator().to(DEVICE)

    # 优化器
    # 生成器使用较低的学习率
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # 判别器使用较高的学习率，帮助它更快收敛
    d_optimizer = optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))

    # 训练循环
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 使用tqdm显示进度条
        for i, (blur_imgs, sharp_imgs) in enumerate(tqdm(dataloader)):
            d_loss, g_total_loss, g_gan_loss, g_l1_loss = train_step(
                blur_imgs, sharp_imgs, generator, discriminator, g_optimizer, d_optimizer)

            # 每50个批次打印一次损失
            if i % 50 == 0:
                print(f"Batch {i}: Gen Loss: {g_total_loss:.4f}, "
                      f"Disc Loss: {d_loss:.4f}, "
                      f"Gen GAN Loss: {g_gan_loss:.4f}, "
                      f"Gen L1 Loss: {g_l1_loss:.4f}")

        # 每个epoch结束后保存示例图像
        test_input, test_target = next(iter(dataloader))
        generate_images(generator, test_input.to(DEVICE), test_target.to(DEVICE),
                        epoch, sample_dir)

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(checkpoint_dir, f'ckpt_{epoch + 1}.pth'))

    return generator


if __name__ == "__main__":
    # 设置您的模糊图像和高清图像目录
    BLUR_DIR = "latent"
    SHARP_DIR = "gallery"

    # 开始训练
    trained_generator = train(BLUR_DIR, SHARP_DIR, epochs=EPOCHS)

    # 保存最终模型
    torch.save(trained_generator.state_dict(), 'fingerprint_enhancement_cgan.pth')
