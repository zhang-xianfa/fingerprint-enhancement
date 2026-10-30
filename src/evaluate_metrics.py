"""
增强结果的无参考图像质量评估

对 results/ 下的增强指纹图逐张计算无参考指标（不需要配对的清晰真值）：
    信息熵      entropy        灰度分布的丰富程度，越高说明细节/灰阶越多
    拉普拉斯方差 laplacian_var  二阶差分响应方差，越大说明高频边缘越清晰
    Tenengrad   tenengrad       Sobel 梯度能量均值，衡量脊线边缘强度
    对比度      contrast       灰度标准差，反映明暗反差
    暗区占比    dark_ratio     低于阈值像素的比例，近似脊线/墨迹的覆盖面积
                               阈值默认用 Otsu 自适应分割，也可用 --dark-threshold 固定

指标只描述增强输出本身的统计特性，不能当作「相对原始检材有所提升」的证据：
原始检材出于合规原因不随仓库提供，且 cGAN 没有配对真值，PSNR/SSIM 不成立。

用法：
    python src/evaluate_metrics.py
    python src/evaluate_metrics.py --input-dir results --output results/metrics.csv
"""

import argparse
import csv
import os
import re

import numpy as np
from PIL import Image


def load_gray(path):
    """读入图像并统一转成 8 位灰度"""
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64)


def calc_entropy(img):
    """灰度直方图的信息熵（bit）"""
    hist = np.histogram(img, bins=256, range=(0, 256))[0].astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def calc_laplacian_var(img):
    """拉普拉斯方差：4 邻域二阶差分，方差越大越清晰"""
    lap = (-4 * img[1:-1, 1:-1] + img[:-2, 1:-1] + img[2:, 1:-1]
           + img[1:-1, :-2] + img[1:-1, 2:])
    return float(lap.var())


def calc_tenengrad(img):
    """Tenengrad：中心差分近似 Sobel，取梯度平方的均值"""
    gx = img[1:-1, 2:] - img[1:-1, :-2]
    gy = img[2:, 1:-1] - img[:-2, 1:-1]
    return float(np.mean(gx ** 2 + gy ** 2))


def calc_contrast(img):
    """对比度：灰度标准差"""
    return float(img.std())


def otsu_threshold(img):
    """Otsu 大津法阈值，把灰度图分成暗（脊线）与亮（背景）两类"""
    hist = np.histogram(img, bins=256, range=(0, 256))[0].astype(np.float64)
    total = hist.sum()
    cum_w = np.cumsum(hist)
    cum_m = np.cumsum(hist * np.arange(256))

    w0 = cum_w[:-1] / total
    w1 = 1.0 - w0
    with np.errstate(divide='ignore', invalid='ignore'):
        mu0 = cum_m[:-1] / cum_w[:-1]
        mu1 = (cum_m[-1] - cum_m[:-1]) / (total - cum_w[:-1])
    between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between)
    return int(np.argmax(between))


def calc_dark_ratio(img, threshold=None):
    """暗区占比：低于阈值的像素比例，近似脊线/墨迹覆盖面积

    threshold 为 None 时用 Otsu 阈值自适应分割。
    返回 (占比, 实际使用的阈值)。
    """
    thr = otsu_threshold(img) if threshold is None else int(threshold)
    return float((img <= thr).mean()), thr


def parse_name(stem):
    """从文件名解析样本号与序号，如 enhanced_10_3 -> ('10', '3')"""
    m = re.match(r'^(?:enhanced_)?(\d+)_(\d+)$', stem)
    if m:
        return m.group(1), m.group(2)
    return stem, ''


def natural_key(filename):
    """按数字大小排序，避免 10 排在 2 前面"""
    return [(0, int(t)) if t.isdigit() else (1, t)
            for t in re.split(r'(\d+)', filename)]


def evaluate_image(path, dark_threshold=None):
    """计算单张图像的全部指标，返回字典"""
    img = load_gray(path)
    dark_ratio, thr = calc_dark_ratio(img, dark_threshold)
    sample, index = parse_name(os.path.splitext(os.path.basename(path))[0])
    return {
        'file': os.path.basename(path),
        'sample': sample,
        'index': index,
        'entropy': calc_entropy(img),
        'laplacian_var': calc_laplacian_var(img),
        'tenengrad': calc_tenengrad(img),
        'contrast': calc_contrast(img),
        'dark_ratio': dark_ratio,
        'dark_threshold': thr,
        'mean_intensity': float(img.mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="对增强结果计算无参考图像质量指标（信息熵 / 清晰度 / 对比度 / 暗区占比）"
    )
    parser.add_argument("--input-dir", default="results",
                        help="增强结果目录（默认 results）")
    parser.add_argument("--output", default=None,
                        help="指标 CSV 输出路径（默认 <input-dir>/metrics.csv）")
    parser.add_argument("--dark-threshold", type=int, default=None,
                        help="暗区占比的固定灰度阈值（0-255），缺省用 Otsu 自适应阈值")
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "metrics.csv")
    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not files:
        raise SystemExit(f"{args.input_dir} 下没有找到图像")
    files.sort(key=natural_key)

    rows = [evaluate_image(os.path.join(args.input_dir, f), args.dark_threshold)
            for f in files]

    # 逐样本打印，方便快速核对
    print(f"共 {len(rows)} 张图像，来自 {len({r['sample'] for r in rows})} 个样本\n")
    print(f"{'文件':<24}{'信息熵':>8}{'拉普拉斯方差':>14}{'Tenengrad':>12}"
          f"{'对比度':>8}{'暗区占比':>10}")
    for r in rows:
        print(f"{r['file']:<24}{r['entropy']:>8.3f}{r['laplacian_var']:>14.2f}"
              f"{r['tenengrad']:>12.2f}{r['contrast']:>8.2f}{r['dark_ratio']:>10.3f}")

    # 汇总均值 / 中位数（写入 CSV 末尾，row_type 区分）
    metric_cols = ['entropy', 'laplacian_var', 'tenengrad', 'contrast',
                   'dark_ratio', 'dark_threshold', 'mean_intensity']
    fields = ['row_type', 'file', 'sample', 'index'] + metric_cols
    with open(output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'row_type': 'image'})
        for label, fn in (('mean', np.mean), ('median', np.median)):
            summary = {'row_type': label, 'file': f'__{label}__', 'sample': '', 'index': ''}
            summary.update({c: float(fn([r[c] for r in rows])) for c in metric_cols})
            writer.writerow(summary)

    print(f"\n汇总（n={len(rows)}）：")
    for c in metric_cols:
        vals = np.array([r[c] for r in rows])
        print(f"  {c:<16} 均值 {vals.mean():>10.3f}   中位数 {np.median(vals):>10.3f}   "
              f"范围 [{vals.min():.3f}, {vals.max():.3f}]")
    print(f"\n指标已写入 {output}")


if __name__ == "__main__":
    main()
