from fingerprint_enhancement import NFIQ  # 假设有NFIQ计算工具
nfiq_before = NFIQ(latent_image)
nfiq_after = NFIQ(enhanced_image)
print(f"NFIQ改进: {nfiq_before} -> {nfiq_after}")

from skimage.metrics import structural_similarity as ssim
ssim_score = ssim(gallery_image, enhanced_image, data_range=255)
print(f"SSIM: {ssim_score:.4f}")

