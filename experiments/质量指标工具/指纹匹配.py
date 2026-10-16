import os
import subprocess
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from multiprocessing import Pool, cpu_count, current_process
from typing import Set, List, Tuple, Dict, Optional, Generator
from functools import lru_cache
import tempfile

# 配置日志
# 设置日志级别为INFO，显示时间戳和消息内容
# 配置日志输出到文件，同时保留控制台输出
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler("app.log"), logging.StreamHandler() ])


def get_temp_prefix():
    pid = current_process().pid
    return f"latent_{pid}_"


# 多线程加速
def parallel_match(args):
    """包装匹配函数以支持多进程"""
    latent_xyt, gallery_xyt = args
    return match_fingerprints(latent_xyt, gallery_xyt)


def get_image_files(dir_path: str):
    """
    获取目录中所有.png指纹图像文件（生成器版本，节省内存）
    参数:
        dir_path: 指纹图像目录路径（字符串类型）
    返回:
        生成器（Generator），逐个产生.png文件名（如"101_1.png"）
        生成器特性：惰性计算，节省内存，适合处理大量文件
    """
    dir_path = Path(dir_path)
    # os.listdir() 列出目录下所有文件和子目录名
    # 参数 dir_path: 目标目录路径
    # 返回: 文件名列表（不包括完整路径）
    for f in os.listdir(dir_path):
        # 检查文件后缀是否为.png（不区分大小写）
        # f.lower(): 将文件名转为小写
        # .endswith('.png'): 判断是否以.png结尾
        if f.lower().endswith('.png'):
            # Path对象使用/运算符拼接路径（跨平台兼容）
            # dir_path: 目录路径（可以是字符串或Path对象）
            # f: 当前文件名
            # 结果 full_path: 完整文件路径（Path对象）
            full_path = dir_path / f
            # 打印调试信息，显示找到的PNG文件完整路径
            # f-string格式化字符串，{full_path}会被实际值替换
            # print(f"找到PNG文件：{full_path}")
            # yield关键字定义生成器函数
            # 每次迭代返回当前文件名f（不包括路径）
            # 函数执行状态会被保留，下次从yield处继续
            yield str(full_path)


def run_command(cmd: str) -> bool:
    """执行命令行指令并检查是否成功"""
    try:
        # 使用subprocess运行命令，shell=True允许执行完整命令字符串
        # check=True表示如果命令执行失败(返回非零退出码)则抛出异常
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        # 捕获命令执行错误，记录错误信息
        logging.error(f"Command failed: {cmd}\nError: {e}")
        return False


def extract_minutiae(img_path: str, output_root: str) -> Optional[str]:
    """
    调用本地 mindtct 工具提取指纹细节点（生成 .xyt 文件）

    参数:
        img_path: 输入的指纹图像路径
        output_root: 输出文件前缀（不含扩展名）

    返回:
        成功生成的.xyt文件路径，失败返回None
    """
    # 1. 获取当前脚本所在目录路径
    script_dir = Path(__file__).parent

    # 2. 构建mindtct工具完整路径
    mindtct_path = script_dir / "mindtct.exe"

    # 3. 检查工具是否存在
    if not mindtct_path.exists():
        logging.error(f"未找到 mindtct 工具: {mindtct_path}")
        return None

    # 4. 检查输入图像是否存在
    if not Path(img_path).exists():
        logging.error(f"输入图像不存在: {img_path}")
        return None

    # 5. 构建命令行指令
    cmd = f'"{mindtct_path}" "{img_path}" "{output_root}"'

    # 6. 执行特征提取命令
    if not run_command(cmd):
        logging.error(f"mindtct 执行失败: {cmd}")
        return None

    # 7. 生成预期输出路径
    xyt_path = f"{output_root}.xyt"

    # 8. 检查文件是否生成成功
    if not Path(xyt_path).exists():
        logging.error(f"特征文件未生成: {xyt_path}")
        return None

    # 9. 验证文件内容有效性
    try:
        with open(xyt_path, 'r') as f:
            # 9.1 读取首行内容
            first_line = f.readline().strip()

            # 9.2 检查是否为空文件
            if not first_line:
                logging.error(f"空特征文件: {xyt_path}")
                os.remove(xyt_path)  # 清理无效文件
                return None

            # 9.3 分割特征点数据
            parts = first_line.split()

            # 9.4 验证特征点格式 (x y theta 三列)
            if len(parts) != 3:
                logging.error(f"无效特征格式: {xyt_path} 首行内容: {first_line}")
                os.remove(xyt_path)
                return None

            # 9.5 验证数值类型
            float(parts[0]), float(parts[1]), float(parts[2])  # 尝试转换为浮点数

    except ValueError:
        logging.error(f"特征点包含非数字内容: {xyt_path}")
        os.remove(xyt_path)
        return None

    except Exception as e:
        logging.error(f"文件读取失败: {str(e)}")
        return None

    # 10. 返回验证通过的文件路径
    return xyt_path


def match_fingerprints(probe_xyt: str, gallery_xyt: str) -> Optional[float]:
    """
    调用 bozorth3 计算两个指纹的匹配分数

    参数:
        probe_xyt: 待匹配指纹的.xyt路径
        gallery_xyt: 库中指纹的.xyt路径

    返回:
        匹配分数（理论上无严格上限，实际常见范围0-100，但可能更高；
        失败时返回None）
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bozorth3_path = os.path.join(script_dir, "bozorth3.exe")

    # ===== 问题1：路径引号嵌套错误 =====
    # 原代码：probe_xyt_normalized = f'"{os.path.normpath(probe_xyt)}"'
    # 会导致路径被双引号包裹两次（如：""path""），应改为：
    probe_xyt_escaped = os.path.normpath(probe_xyt).replace('"', '""')  # 仅转义内部引号
    gallery_xyt_escaped = os.path.normpath(gallery_xyt).replace('"', '""')

    # ===== 问题2：命令构造方式不安全 =====
    # 原代码使用字符串拼接，存在空格和特殊字符风险
    # 应改用参数列表形式，且显式处理路径引号
    cmd = [
        bozorth3_path,
        f'"{probe_xyt_escaped}"',  # 直接包裹引号
        f'"{gallery_xyt_escaped}"'
    ]

    try:
        # ===== 问题3：未验证输入文件存在性 =====
        if not os.path.exists(probe_xyt) or os.path.getsize(probe_xyt) == 0:
            logging.warning(f"无效探测文件: {probe_xyt}")
            return None
        if not os.path.exists(gallery_xyt) or os.path.getsize(gallery_xyt) == 0:
            logging.warning(f"无效库文件: {gallery_xyt}")
            return None

        # ===== 问题4：shell=True 可能导致意外行为 =====
        # 使用参数列表时，应设置 shell=False（Windows 下仍需 True）
        # 但需处理路径中的空格，改为显式调用 cmd.exe
        result = subprocess.run(
            " ".join(cmd),  # 转换为字符串命令
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='gbk',  # Windows 控制台默认编码为 GBK，非 UTF-8
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW  # 隐藏命令行窗口（仅 Windows）
        )

        # ===== 问题5：未处理非数字输出 =====
        output = result.stdout.strip()
        if not output.replace('.', '', 1).isdigit():  # 允许小数点和数字
            logging.warning(f"bozorth3 输出格式异常: {output}")
            return None

        return float(output)

    except subprocess.CalledProcessError as e:
        # ===== 问题6：未记录详细错误信息 =====
        error_msg = f"""
        bozorth3 执行失败！
        命令: {e.cmd}
        退出码: {e.returncode}
        错误输出: {e.stderr}
        """
        logging.error(error_msg)
        return None

    except subprocess.TimeoutExpired:
        logging.error("bozorth3 执行超时（超过 10 秒）")
        return None

    except Exception as e:
        logging.error(f"未知错误: {str(e)}", exc_info=True)  # 记录堆栈跟踪
        return None


@lru_cache(maxsize=1000)  # 缓存最多1000个结果
def cached_extract_minutiae(img_path: str, output_path: str) -> Optional[str]:
    """
    带缓存的细节点提取函数

    参数:
        img_path: 指纹图像路径
        output_path: 输出.xyt文件路径

    返回:
        生成的.xyt文件路径（失败返回None）
    """
    # 如果输出文件已存在，直接返回路径
    if os.path.exists(output_path):
        logging.debug(f"使用缓存结果: {output_path}")
        return output_path
    # 否则调用原始提取函数
    return extract_minutiae(img_path, output_path)


def calculate_rank_n(latent_dir: str,gallery_dir: str,n: int,temp_dir: str) -> float:
    """
    计算 Rank-n 识别率（精确匹配文件名，区分如 26_0 和 26_1）
    参数:
        latent_dir: 潜在指纹图像目录路径（如 "data/latent"）
        gallery_dir: 库指纹图像目录路径（如 "data/gallery"）
        n: Rank阈值（如 5 或 20）
        temp_dir: 临时文件存储目录路径（如 "temp"）
    返回:
        Rank-n 识别率（百分比，0.0~100.0）
    """
    # 1. 创建临时目录（如果不存在）
    os.makedirs(temp_dir, exist_ok=True)

    # 2. 初始化临时文件集合（用于记录生成的.xyt文件路径）
    temp_files: Set[str] = set()

    # 3. 获取潜在指纹和库指纹的文件列表（仅.png文件）
    latent_files = list(get_image_files(latent_dir))  # 如 ["26_0.png", "26_1.png"]
    gallery_files = list(get_image_files(gallery_dir))  # 如 ["26_0.png", "26_1.png", "101.png"]

    # 4. 检查输入目录是否为空
    if not latent_files or not gallery_files:
        logging.warning("输入目录为空！")
        return 0.0  # 返回0%识别率

    # 5. 初始化正确匹配计数器
    correct_matches = 0

    # 6. 预处理：提取所有库指纹的特征（.xyt文件）
    gallery_xyt_map: Dict[str, str] = {}  # 格式: {"26_0": "temp/gallery_26_0.xyt", ...}
    for gallery_file in gallery_files:
        # 6.1 获取库指纹的ID（不含扩展名，如 "26_0"）
        gallery_id = Path(gallery_file).stem

        # 6.2 生成对应的.xyt文件路径（如 "temp/gallery_26_0.xyt"）
        gallery_xyt = os.path.join(temp_dir, f"gallery_{gallery_id}.xyt")

        # 6.3 调用带缓存的细节点提取函数
        if cached_extract_minutiae(
                # os.path.join(gallery_dir, gallery_file),  # 库指纹完整路径
                gallery_file,
                gallery_xyt  # 输出.xyt路径
        ):
            gallery_xyt_map[gallery_id] = gallery_xyt  # 记录成功提取的特征文件
            temp_files.add(gallery_xyt)  # 加入临时文件集合

    # 7. 处理每个潜在指纹
    for latent_file in latent_files:
        # 7.1 获取潜在指纹的ID（如 "26_0"）
        latent_id = Path(latent_file).stem
        pid = current_process().pid
        # 7.2 生成对应的.xyt文件路径（如 "temp/latent_26_0.xyt"）
        latent_xyt = os.path.join(temp_dir, f"latent_{pid}_{latent_id}.xyt")

        # 7.3 提取潜在指纹特征（跳过失败的文件）
        if not cached_extract_minutiae(
                # os.path.join(latent_dir, latent_file),  # 潜在指纹完整路径
                latent_file,
                latent_xyt  # 输出.xyt路径
        ):
            continue

        temp_files.add(latent_xyt)  # 记录临时文件

        # 7.4 并行匹配：计算潜在指纹与所有库指纹的匹配分数
        scores: List[Tuple[str, float]] = []  # 格式: [("26_0", 95.3), ("26_1", 92.1), ...]

        # 7.5 准备多进程参数（每个任务是对比一个库指纹）
        args = [
            (latent_xyt, gallery_xyt)
            for gallery_xyt in gallery_xyt_map.values()
        ]

        # 7.6 使用多进程池并行匹配
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            # 7.6.1 并行执行匹配任务
            for gallery_id, score in zip(
                    gallery_xyt_map.keys(),  # 库指纹ID列表（如 ["26_0", "26_1"]）
                    pool.imap(parallel_match, args)  # 并行调用 match_fingerprints
            ):
                if score is not None:  # 忽略失败的匹配
                    scores.append((gallery_id, score))

        # 7.7 计算Rank-n命中
        if scores:  # 如果有有效匹配结果
            # 7.7.1 按分数降序排序（高分在前）
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            # 7.7.2 取前n名的库指纹ID
            top_n_ids = [x[0] for x in sorted_scores[:n]]  # 如 ["26_0", "101"]

            # 7.7.3 检查潜在指纹的真实ID是否在前n名中（精确匹配）
            is_hit = latent_id in top_n_ids  # 如 "26_0" in ["26_0", "101"] -> True

            if is_hit:
                correct_matches += 1  # 命中计数

    # 8. 计算最终识别率（百分比）
    recognition_rate = (
        (correct_matches / len(latent_files)) * 100
        if latent_files
        else 0.0
    )

    # 9. 记录日志
    logging.info(f"Rank-{n} 识别率: {recognition_rate:.2f}%")

    # 10. 返回结果（临时文件由外部清理）
    return recognition_rate


def plot_results(rank_5: float, rank_20: float, output_file: str = "results.png"):
    """绘制Rank-5和Rank-20识别率柱状图"""
    plt.figure(figsize=(8, 5))  # 创建8x5英寸的图形
    # 绘制柱状图，设置不同颜色
    bars = plt.bar(["Rank-5", "Rank-20"], [rank_5, rank_20], color=["#4C72B0", "#DD8452"])
    plt.title("Fingerprint Recognition Rates", pad=20)  # 设置标题
    plt.ylabel("Recognition Rate (%)")  # 设置Y轴标签
    plt.ylim(0, 100)  # 设置Y轴范围为0-100

    # 在每个柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f"{height:.1f}%", ha="center", va="bottom")

    # 保存图表并显示
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    # 设置指纹图像目录路径
    LATENT_DIR = r"E:\Program Files (x86)\pythonProject\数学建模\省数学建模\data\latent"
    GALLERY_DIR = r"E:\Program Files (x86)\pythonProject\数学建模\省数学建模\data\gallery"
    # 生成唯一临时目录（系统自动管理）
    temp_dir = tempfile.mkdtemp(prefix="fp_match_")
    logging.info("Calculating Rank-5...")
    rank_5 = calculate_rank_n(LATENT_DIR, GALLERY_DIR, 5, temp_dir)
    logging.info("Calculating Rank-20...")
    rank_20 = calculate_rank_n(LATENT_DIR, GALLERY_DIR, 20, temp_dir)

    print(f"Rank-5: {rank_5:.2f}%, Rank-20: {rank_20:.2f}%")
    # 输出结果
    print(f"\n{' Results ':=^40}")  # 格式化输出分隔线
    print(f"Rank-5 Recognition Rate:  {rank_5:.2f}%")
    print(f"Rank-20 Recognition Rate: {rank_20:.2f}%")
    # 绘制结果图表
    plot_results(rank_5, rank_20)
