import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


# 定义生成器
class Generator(nn.Module):
    """
    基于CNN的图像生成器网络，实现图像到图像的转换
    采用Encoder-Decoder结构，通过卷积和反卷积操作改变图像尺寸
    """

    def __init__(self):
        super(Generator, self).__init__()  # 调用父类构造函数初始化模块

        # 编码器部分（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # 输入1通道灰度图
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 新增深度层
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # 解码器部分（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # 输出1通道灰度图
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        """前向传播函数，定义数据流经网络的路径"""
        return self.decoder(x)  # 按顺序通过所有层

# 定义判别器
class Discriminator(nn.Module):
    """
    条件对抗网络判别器（Conditional GAN Discriminator）
    功能：判断输入的「条件图像+生成图像」是否为真实样本
    结构：全卷积网络，输出单通道概率值（0-1之间）
    """

    def __init__(self):
        super(Discriminator, self).__init__()  # 调用父类构造函数初始化模块

        # 定义判别器网络结构（全卷积架构，无全连接层）
        self.model = nn.Sequential(
            # 输入通道数=2（条件图1+输入图1）
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),  # 添加Dropout

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 新增深度层
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # 输出单通道概率图
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        前向传播函数
        参数：
            x: 条件图像（如真实图像的标签、语义图等，3通道）
            y: 生成图像或真实图像（3通道）
        返回：
            判别概率值（单通道标量，越接近1越真实）
        """
        # 在通道维度（dim=1）拼接条件图像和输入图像
        # 例如：x.shape=(batch, 3, H, W), y.shape=(batch, 3, H, W) → 拼接后为(batch, 6, H, W)
        x = torch.cat([x, y], dim=1)

        # 将拼接后的特征图输入网络，输出判别结果
        return self.model(x)


# 自定义数据集（明确灰度处理）
class FingerprintDataset(Dataset):
    """
    灰度图像专用数据集类
    修改说明：
    1. 明确使用L模式读取单通道灰度图
    2. 添加数据增强策略
    """

    def __init__(self, latent_dir, gallery_dir, transform=None):
        # 初始化时添加随机水平翻转和颜色抖动增强
        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 确保单通道输出
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),  # 数据增强
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.3),  # 对比度抖动
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 单通道归一化
        ])

        # 初始化路径参数
        self.latent_dir = latent_dir
        self.gallery_dir = gallery_dir
        self.image_names = [f for f in os.listdir(latent_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        latent_path = os.path.join(self.latent_dir, self.image_names[idx])
        gallery_path = os.path.join(self.gallery_dir, self.image_names[idx])

        # 保持图像为单通道
        latent_img = Image.open(latent_path).convert('L')  # 明确转换为灰度模式
        gallery_img = Image.open(gallery_path).convert('L')

        # 应用相同的数据增强
        seed = torch.randint(0, 2 ** 32, ())  # 同步随机种子
        torch.manual_seed(seed)
        latent_img = self.base_transform(latent_img)
        torch.manual_seed(seed)  # 保证相同变换
        gallery_img = self.base_transform(gallery_img)

        return latent_img, gallery_img
# 训练函数
def train_cgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化网络
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 优化器配置（差异化学习率）
    # 生成器使用较低学习率，判别器使用较高学习率以平衡训练
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.999))

    # 动态调整学习率
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=0.5, patience=5)

    # 损失函数配置
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    lambda_L1 = 100  # 根据实验效果可调整至50-150范围

    # 训练参数
    num_epochs = 30
    batch_size = 16  # 适当增大batch_size
    dataset = FingerprintDataset('latent', 'gallery')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 训练循环改进策略
    for epoch in range(num_epochs):
        G.train()
        D.train()
        running_loss_G = 0.0
        running_loss_D = 0.0

        for i, (latent, gallery) in enumerate(dataloader):
            latent = latent.to(device)
            gallery = gallery.to(device)

            # --- 训练判别器 ---
            optimizer_D.zero_grad()

            # 真实样本损失
            pred_real = D(latent, gallery)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # 生成样本损失
            with torch.no_grad():
                fake = G(latent)
            pred_fake = D(latent, fake.detach())  # 阻断生成器梯度
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            # 总判别器损失
            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # --- 训练生成器 ---
            # 每2个step训练一次生成器以平衡训练
            if i % 2 == 0:
                optimizer_G.zero_grad()

                # 生成图像
                fake = G(latent)

                # 对抗损失
                pred_fake = D(latent, fake)
                g_loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

                # L1重建损失
                g_loss_L1 = criterion_L1(fake, gallery)

                # 总生成器损失
                g_loss = g_loss_GAN + lambda_L1 * g_loss_L1
                g_loss.backward()
                optimizer_G.step()

            # 统计损失
            running_loss_D += d_loss.item()
            running_loss_G += g_loss.item() if i % 2 == 0 else 0

            # 每20个batch输出日志
            if i % 20 == 0:
                print(f"[Epoch {epoch + 1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item() if i % 2 == 0 else 0:.4f}")


        # 计算epoch平均损失
        epoch_loss_G = running_loss_G / (len(dataloader) // 2)
        epoch_loss_D = running_loss_D / len(dataloader)

        # 调整学习率
        scheduler_G.step(epoch_loss_G)
        scheduler_D.step(epoch_loss_D)


    # 最终模型保存
    torch.save(G.state_dict(), 'generator_final.pth')
    torch.save(D.state_dict(), 'discriminator_final.pth')


if __name__ == '__main__':
    train_cgan()
