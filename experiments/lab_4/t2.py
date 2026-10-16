import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


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


