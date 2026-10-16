import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义文件夹路径
LATENT_DIR = "gallery"  # 潜在指纹文件夹
GALLERY_DIR = "latent"  # 库指纹文件夹

# 初始阈值（可根据经验或初步分析设定）
INITIAL_THRESHOLD = 20  # 初始假设阈值


def extract_minutiae(image_path, output_root):
    """提取指纹细节点"""
    try:
        subprocess.run(["mindtct.exe", image_path, output_root], check=True)
        print(f"细节点提取成功: {image_path} -> {output_root}.xyt")
    except subprocess.CalledProcessError as e:
        print(f"细节点提取失败: {image_path}, 错误: {e}")


def match_fingerprints(probe_xyt, gallery_xyt, threshold=INITIAL_THRESHOLD):
    """计算指纹匹配分数，并根据阈值过滤"""
    try:
        result = subprocess.run(["bozorth3.exe", probe_xyt, gallery_xyt], capture_output=True, text=True, check=True)
        score = float(result.stdout.strip())
        return score if score >= threshold else 0.0  # 低于阈值返回0
    except subprocess.CalledProcessError as e:
        print(f"匹配失败: {probe_xyt} vs {gallery_xyt}, 错误: {e}")
        return 0.0


def calculate_rank_n(latent_dir, gallery_dir, n_values=[5, 20], initial_threshold=INITIAL_THRESHOLD):
    """计算Rank-n识别率，先使用初始阈值，再通过ROC优化"""
    rank_results = {n: {"correct": 0, "total": 0} for n in n_values}
    all_scores = []
    y_true = []  # 真实标签（1=匹配，0=不匹配）
    y_score = []  # 匹配分数

    # 第一步：使用初始阈值统计匹配结果
    for latent_file in os.listdir(latent_dir):
        if not latent_file.endswith(".png"):
            continue

        latent_name = os.path.splitext(latent_file)[0]
        latent_path = os.path.join(latent_dir, latent_file)
        output_root = os.path.join(latent_dir, latent_name)

        # 提取潜在指纹细节点
        extract_minutiae(latent_path, output_root)
        probe_xyt = f"{output_root}.xyt"
        if not os.path.exists(probe_xyt):
            continue

        # 仅匹配同名库指纹
        gallery_file = f"{latent_name}.png"
        if gallery_file not in os.listdir(gallery_dir):
            continue

        gallery_path = os.path.join(gallery_dir, gallery_file)
        gallery_root = os.path.join(gallery_dir, latent_name)
        extract_minutiae(gallery_path, gallery_root)
        gallery_xyt = f"{gallery_root}.xyt"

        # 计算匹配分数（应用初始阈值）
        score = match_fingerprints(probe_xyt, gallery_xyt, initial_threshold)
        all_scores.append((latent_name, gallery_file, score))
        y_true.append(1)  # 假设同名文件为真实匹配
        y_score.append(score)

        # 统计Rank-n
        for n in n_values:
            rank_results[n]["total"] += 1
            if gallery_file == f"{latent_name}.png" and score > 0:  # 分数>0表示通过阈值
                rank_results[n]["correct"] += 1

    # 第二步：通过ROC曲线优化阈值
    if len(y_true) > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        optimal_idx = np.argmax(tpr - fpr)  # Youden指数最大
        optimal_threshold = thresholds[optimal_idx]
        print(f"初始阈值: {initial_threshold}, 优化后阈值: {optimal_threshold:.1f}")

        # 重新计算识别率（使用优化后的阈值）
        for n in rank_results:
            rank_results[n]["correct"] = 0
            rank_results[n]["total"] = 0

        for latent_name, gallery_file, score in all_scores:
            for n in n_values:
                rank_results[n]["total"] += 1
                if gallery_file == f"{latent_name}.png" and score >= optimal_threshold:
                    rank_results[n]["correct"] += 1

    # 计算最终识别率
    for n in rank_results:
        total = rank_results[n]["total"]
        correct = rank_results[n]["correct"]
        rank_results[n]["rate"] = (correct / total) * 100 if total > 0 else 0.0

    return rank_results, all_scores, np.array(y_true), np.array(y_score)


if __name__ == "__main__":
    # 初始运行（使用初始阈值）
    rank_results, all_scores, y_true, y_score = calculate_rank_n(LATENT_DIR, GALLERY_DIR)

    # 打印识别率结果
    for n, data in rank_results.items():
        print(f"Rank-{n} 识别率: {data['rate']:.2f}% (正确: {data['correct']}, 总数: {data['total']})")

    # 可视化分数分布
    scores = [score for _, _, score in all_scores]
    plt.hist(scores, bins=20)
    plt.xlabel("匹配分数")
    plt.ylabel("频次")
    plt.title("bozorth3 分数分布")
    plt.show()

    # 可视化ROC曲线
    if len(y_true) > 0:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label='ROC曲线')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('ROC曲线分析')
        plt.legend()
        plt.show()
# inf 表无限大，即所有匹配分数相同：如果所有指纹对的匹配分数完全相同（如全为0或全为100），ROC曲线无法计算有效阈值。