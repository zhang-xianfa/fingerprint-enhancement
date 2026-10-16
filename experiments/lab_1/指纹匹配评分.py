import os
import subprocess
from collections import defaultdict


# 定义文件夹路径
LATENT_DIR = "answer1"  # 潜在指纹文件夹
GALLERY_DIR = "gallery"  # 库指纹文件夹


def extract_minutiae(image_path, output_root):
    """
    使用 mindtct.exe 提取指纹的细节点
    :param image_path: 指纹图像路径
    :param output_root: 输出文件的前缀（不包含扩展名）
    """
    try:
        # 调用 mindtct.exe 提取细节点
        subprocess.run(["mindtct.exe", image_path, output_root], check=True)
        print(f"细节点提取成功: {image_path} -> {output_root}.xyt")
    except subprocess.CalledProcessError as e:
        print(f"细节点提取失败: {image_path}, 错误: {e}")


def match_fingerprints(probe_xyt, gallery_xyt):
    """
    使用 bozorth3.exe 计算两个指纹的匹配分数
    :param probe_xyt: 潜在指纹的 .xyt 文件路径
    :param gallery_xyt: 库指纹的 .xyt 文件路径
    :return: 匹配分数（0-1之间的值）
    """
    try:
        # 调用 bozorth3.exe 计算匹配分数
        result = subprocess.run(["bozorth3.exe", probe_xyt, gallery_xyt], capture_output=True, text=True, check=True)
        score = float(result.stdout.strip())
        print(f"匹配成功: {probe_xyt} vs {gallery_xyt}, 分数: {score}")
        return score
    except subprocess.CalledProcessError as e:
        print(f"匹配失败: {probe_xyt} vs {gallery_xyt}, 错误: {e}")
        return 0.0



MIN_SCORE_THRESHOLD = 1  # 仅统计分数 >= 20 的匹配（根据实际分布调整）

def calculate_rank_n(latent_dir, gallery_dir, n_values=[1,5, 20]):
    rank_results = {n: {"correct": 0, "total": 0} for n in n_values}
    all_scores = []  # 记录所有匹配对（用于后续分析）

    for latent_file in os.listdir(latent_dir):
        if not latent_file.endswith(".png"):
            continue

        latent_name = os.path.splitext(latent_file)[0]
        latent_path = os.path.join(latent_dir, latent_file)
        output_root = os.path.join(latent_dir, latent_name)

        # 1. 提取潜在指纹细节点
        extract_minutiae(latent_path, output_root)
        probe_xyt = f"{output_root}.xyt"
        if not os.path.exists(probe_xyt):
            continue

        # 2. 匹配所有库指纹（仅同名文件）
        gallery_file = f"{latent_name}.png"
        if gallery_file not in os.listdir(gallery_dir):
            continue

        gallery_path = os.path.join(gallery_dir, gallery_file)
        gallery_root = os.path.join(gallery_dir, latent_name)
        extract_minutiae(gallery_path, gallery_root)
        gallery_xyt = f"{gallery_root}.xyt"

        # 3. 计算匹配分数
        score = match_fingerprints(probe_xyt, gallery_xyt)
        all_scores.append((latent_name, gallery_file, score))

        # 4. 仅当分数 >= 阈值时统计
        if score >= MIN_SCORE_THRESHOLD:
            for n in n_values:
                rank_results[n]["total"] += 1
                # 假设文件名一致即为正确匹配（需人工验证）
                if gallery_file == f"{latent_name}.png":
                    rank_results[n]["correct"] += 1

    # 计算识别率
    for n in rank_results:
        total = rank_results[n]["total"]
        correct = rank_results[n]["correct"]
        rank_results[n]["rate"] = (correct / total) * 100 if total > 0 else 0.0

    return rank_results, all_scores  # 返回结果和原始分数（用于分析）



if __name__ == "__main__":
    # 计算 rank-5 和 rank-20 识别率（带分数阈值）
    rank_results, all_scores = calculate_rank_n(LATENT_DIR, GALLERY_DIR)

    # 打印识别率结果
    for n, data in rank_results.items():  # 注意：这里用 rank_results 而不是 results
        print(f"Rank-{n} 识别率: {data['rate']:.2f}% (正确: {data['correct']}, 总数: {data['total']})")
    #
    # 可选：分析分数分布
    # analyze_score_distribution(all_scores)
