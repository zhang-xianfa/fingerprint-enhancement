import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2  # 导入OpenCV库

class FingerprintDataset(Dataset):
    def __init__(self, latent_dir, gallery_dir):
        """
        latent_dir: 预处理后的latent图像(.npy)路径
        gallery_dir: 对应的gallery图像(.npy)路径
        """
        self.latent_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.npy')])
        self.gallery_files = sorted([f for f in os.listdir(gallery_dir) if f.endswith('.npy')])
        self.latent_dir = latent_dir
        self.gallery_dir = gallery_dir

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # 加载latent和gallery图像
        latent = np.load(os.path.join(self.latent_dir, self.latent_files[idx]))
        gallery = np.load(os.path.join(self.gallery_dir, self.gallery_files[idx]))

        # 转换为PyTorch张量并增加通道维度
        latent = torch.FloatTensor(latent).unsqueeze(0)  # shape: [1, 256, 256]
        gallery = torch.FloatTensor(gallery).unsqueeze(0)

        return latent, gallery


# 示例使用
latent_dir = "processed_latent"
gallery_dir = "processed_gallery"
dataset = FingerprintDataset(latent_dir, gallery_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()  # 输出范围[-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),  # 输入是latent+generated/gallery
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0), nn.Sigmoid()  # 输出每个patch的真伪概率
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)  # 拼接条件(latent)和生成/真实图像
        return self.model(xy)
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

# 损失函数
criterion_adv = nn.BCELoss()  # 对抗损失
criterion_l1 = nn.L1Loss()     # 重构损失
lambda_l1 = 100               # L1损失的权重

# 优化器
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

# 损失函数
criterion_adv = nn.BCELoss()  # 对抗损失
criterion_l1 = nn.L1Loss()     # 重构损失
lambda_l1 = 100               # L1损失的权重

# 优化器
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

def generate_enhanced_image(model, latent_path):
    latent = np.load(latent_path)
    latent = torch.FloatTensor(latent).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced = G(latent)
    enhanced = enhanced.squeeze().cpu().numpy()
    enhanced = (enhanced + 1) * 127.5  # 从[-1,1]映射到[0,255]
    return enhanced.astype(np.uint8)

# 示例生成
sample_latent = "processed_latent/26_3.npy"
enhanced_image = generate_enhanced_image(G, sample_latent)
cv2.imwrite("enhanced_26_3.png", enhanced_image)

