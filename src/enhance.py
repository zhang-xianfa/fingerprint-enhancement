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
import torch.nn.functional as F

# 配置参数
IMG_SIZE = 512
CHANNELS = 1
BATCH_SIZE = 4
EPOCHS = 100
LAMBDA = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 生成器架构（保持原始结构）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = DownBlock(1, 64, batchnorm=False)  # 512→256
        self.enc2 = DownBlock(64, 128)  # 256→128
        self.enc3 = DownBlock(128, 256)  # 128→64
        self.enc4 = DownBlock(256, 512)  # 64→32
        self.enc5 = DownBlock(512, 512)  # 32→16
        self.enc6 = DownBlock(512, 512)  # 16→8
        self.enc7 = DownBlock(512, 512)  # 8→4

        # 解码器（修正输入通道数）
        self.dec1 = UpBlock(512, 512)  # 输入512 → 输出512 (4→8)
        self.dec2 = UpBlock(1024, 512)  # 输入1024 → 输出512 (8→16)
        self.dec3 = UpBlock(1024, 512)  # 输入1024 → 输出512 (16→32)
        self.dec4 = UpBlock(1024, 256)  # 输入1024 → 输出256 (32→64)
        self.dec5 = UpBlock(512, 128)  # 输入512 → 输出128 (64→128)
        self.dec6 = UpBlock(256, 64)  # 输入256 → 输出64 (128→256)
        self.dec7 = UpBlock(128, 32)  # 输入128 → 输出32 (256→512)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码过程
        e1 = self.enc1(x)  # 64,256x256
        e2 = self.enc2(e1)  # 128,128x128
        e3 = self.enc3(e2)  # 256,64x64
        e4 = self.enc4(e3)  # 512,32x32
        e5 = self.enc5(e4)  # 512,16x16
        e6 = self.enc6(e5)  # 512,8x8
        e7 = self.enc7(e6)  # 512,4x4

        # 解码过程
        d1 = self.dec1(e7)  # 512,8x8
        d1 = torch.cat([d1, e6], 1)  # 1024,8x8

        d2 = self.dec2(d1)  # 512,16x16
        d2 = torch.cat([d2, e5], 1)  # 1024,16x16

        d3 = self.dec3(d2)  # 512,32x32
        d3 = torch.cat([d3, e4], 1)  # 1024,32x32

        d4 = self.dec4(d3)  # 256,64x64
        d4 = torch.cat([d4, e3], 1)  # 512,64x64

        d5 = self.dec5(d4)  # 128,128x128
        d5 = torch.cat([d5, e2], 1)  # 256,128x128

        d6 = self.dec6(d5)  # 64,256x256
        d6 = torch.cat([d6, e1], 1)  # 128,256x256

        d7 = self.dec7(d6)  # 32,512x512

        return self.final(d7)  # 1,512x512


# 判别器架构（保持原始结构）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool2d(3, 2, 1)
        self.discriminators = nn.ModuleList([
            self._build_discriminator(),
            self._build_discriminator(),
            self._build_discriminator()
        ])

    def _build_discriminator(self):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(2, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1)),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x_pairs):  # 修改为接收单输入
        outputs = []
        current_x = x_pairs
        for i in range(3):
            if i > 0:
                current_x = self.downsample(current_x)
            outputs.append(self.discriminators[i](current_x))
        return outputs


# 公共模块
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if batchnorm: layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ]
        if dropout: layers.insert(2, nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 数据加载
class PairedDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        self.blur_paths = sorted([os.path.join(blur_dir, f) for f in os.listdir(blur_dir)])
        self.sharp_paths = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir)])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
            ], p=0.5),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur = cv2.imread(self.blur_paths[idx], 0)
        sharp = cv2.imread(self.sharp_paths[idx], 0)

        # 动态调整尺寸
        blur = cv2.resize(blur, (IMG_SIZE, IMG_SIZE))
        sharp = cv2.resize(sharp, (IMG_SIZE, IMG_SIZE))

        return self.transform(blur), self.transform(sharp)


# 训练系统
class CGANTrainer:
    def __init__(self, blur_dir, sharp_dir, batch_size=BATCH_SIZE):
        self.dataset = PairedDataset(blur_dir, sharp_dir)
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size, shuffle=True)

        self.G = Generator().to(DEVICE)
        self.D = Discriminator().to(DEVICE)

        self.opt_G = optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.scheduler_G = optim.lr_scheduler.StepLR(self.opt_G, 30, 0.5)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.opt_D, 30, 0.5)
        self.sample_dir = "training_samples"
        os.makedirs(self.sample_dir, exist_ok=True)
        self.model_dir = "saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.warmup_epochs = 2
    def _save_sample_images(self, epoch):
        # 设置为评估模式
        self.G.eval()

        # 获取一个测试batch
        with torch.no_grad():
            blur, sharp = next(iter(self.loader))
            blur = blur.to(DEVICE)
            gen_imgs = self.G(blur)

        # 准备图片数据
        blur = blur.cpu()
        gen_imgs = gen_imgs.cpu()
        sharp = sharp.cpu()

        # 反归一化 [-1,1] -> [0,1]
        reverse_norm = transforms.Normalize(mean=(-0.5,), std=(0.5,))

        # 创建对比图（实际 batch 可能小于设定值）
        n_show = min(self.batch_size, blur.size(0))
        fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
        plt.suptitle(f'Epoch {epoch}', fontsize=20)

        for i in range(n_show):
            # 输入图像（模糊）
            inp = reverse_norm(blur[i]).squeeze().numpy()
            # 生成图像
            gen = reverse_norm(gen_imgs[i]).squeeze().numpy()
            # 真实图像（清晰）
            real = reverse_norm(sharp[i]).squeeze().numpy()

            # 绘制图像
            axes[i, 0].imshow(inp, cmap='gray')
            axes[i, 0].set_title("Input (Blur)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gen, cmap='gray')
            axes[i, 1].set_title("Generated")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(real, cmap='gray')
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')

        # 保存图片
        plt.savefig(os.path.join(self.sample_dir, f"epoch_{epoch:03d}.png"))
        plt.close()

    def _gen_loss(self, fake_preds, gen_imgs, real_imgs):
        gan_loss = sum(F.binary_cross_entropy_with_logits(p, torch.ones_like(p)) for p in fake_preds)
        l1_loss = F.l1_loss(gen_imgs, real_imgs)
        return gan_loss + LAMBDA * l1_loss, gan_loss, l1_loss

    def _disc_loss(self, real_preds, fake_preds):
        loss = 0
        for rp, fp in zip(real_preds, fake_preds):
            real_loss = F.binary_cross_entropy_with_logits(rp, torch.ones_like(rp))
            fake_loss = F.binary_cross_entropy_with_logits(fp, torch.zeros_like(fp))
            loss += (real_loss + fake_loss) / len(real_preds)
        return loss

    def train_step(self, blur_imgs, sharp_imgs):
        blur_imgs = blur_imgs.to(DEVICE)
        sharp_imgs = sharp_imgs.to(DEVICE)

        # ===== 训练判别器（仅在预热期后执行）=====
        if self.current_epoch >= self.warmup_epochs:
            self.opt_D.zero_grad()

            with torch.no_grad():
                gen_imgs = self.G(blur_imgs)

            real_pairs = torch.cat([blur_imgs, sharp_imgs], dim=1)
            fake_pairs = torch.cat([blur_imgs, gen_imgs], dim=1)

            real_preds = self.D(real_pairs)
            fake_preds = self.D(fake_pairs)

            d_loss = self._disc_loss(real_preds, fake_preds)
            d_loss.backward()
            self.opt_D.step()
        else:
            d_loss = torch.tensor(0.0)  # 预热期不计算判别器损失

        # ===== 训练生成器 =====
        self.opt_G.zero_grad()

        # 生成图像
        gen_imgs = self.G(blur_imgs)

        # 计算损失（预热期仅使用L1损失）
        if self.current_epoch >= self.warmup_epochs:
            fake_pairs = torch.cat([blur_imgs, gen_imgs], dim=1)
            fake_preds = self.D(fake_pairs)
            g_loss, g_gan, g_l1 = self._gen_loss(fake_preds, gen_imgs, sharp_imgs)
        else:
            g_l1 = F.l1_loss(gen_imgs, sharp_imgs) * LAMBDA
            g_loss = g_l1
            g_gan = torch.tensor(0.0)

        g_loss.backward()
        self.opt_G.step()

        return {
            'd_loss': d_loss.item(),
            'g_total': g_loss.item(),
            'g_gan': g_gan.item(),
            'g_l1': g_l1.item()
        }

    def train(self, epochs):
        for epoch in range(epochs):
            self.current_epoch = epoch  # 记录当前epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 设置训练模式
            self.G.train()
            self.D.train() if epoch >= self.warmup_epochs else self.D.eval()

            # 训练循环
            for batch, (blur, sharp) in enumerate(tqdm(self.loader)):
                metrics = self.train_step(blur, sharp)

                # 打印日志
                if batch % 50 == 0:
                    log = f"[Batch {batch}]"
                    if epoch >= self.warmup_epochs:
                        log += f" D: {metrics['d_loss']:.4f} | G: {metrics['g_total']:.4f}"
                        log += f" (GAN:{metrics['g_gan']:.4f} L1:{metrics['g_l1']:.4f})"
                    else:
                        log += f" Pre-train G | L1: {metrics['g_l1']:.4f}"
                    print(log)

            # 保存模型和样本
            self._save_sample_images(epoch + 1)
            model_path = os.path.join(self.model_dir, f"generator_epoch_{epoch + 1}.pth")
            torch.save(self.G.state_dict(), model_path)

            # 更新学习率
            if epoch >= self.warmup_epochs:
                self.scheduler_D.step()
            self.scheduler_G.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="潜在指纹增强：多尺度 PatchGAN 条件生成对抗网络"
    )
    parser.add_argument("--blur-dir", required=True,
                        help="模糊（待增强）指纹图像目录")
    parser.add_argument("--sharp-dir", required=True,
                        help="清晰（参考/真值）指纹图像目录，与 --blur-dir 文件名一一对应")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"训练轮数（默认 {EPOCHS}）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"批大小（默认 {BATCH_SIZE}）")
    parser.add_argument("--out", default="final_generator.pth",
                        help="最终生成器权重保存路径")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    trainer = CGANTrainer(args.blur_dir, args.sharp_dir, batch_size=args.batch_size)
    trainer.train(args.epochs)
    torch.save(trainer.G.state_dict(), args.out)
    print(f"训练完成，生成器权重已保存至 {args.out}")