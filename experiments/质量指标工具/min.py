"""
该文件是一个指纹细节点（Minutiae）特征文件，
记录了从指纹图像中提取的关键细节点信息（如脊线端点、分叉点），用于指纹识别或匹配。
"""
import matplotlib.pyplot as plt


def read_min_file(file_path):
    """
    安全读取.min文件，自动跳过格式错误行
    """
    minutiae_data = []
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith(("Image", "Minutiae")):
                continue

            parts = line.split(':')
            if len(parts) < 6:
                print(f"警告：第 {line_num} 行缺少必要字段，已跳过")
                continue

            try:
                # 解析基础信息
                minutiae = {
                    'id': int(parts[0].strip()),
                    'x': int(parts[1].split(',')[0].strip()),
                    'y': int(parts[1].split(',')[1].strip()),
                    'angle': int(parts[2].strip()),
                    'quality': float(parts[3].strip()),
                    'type': parts[4].strip(),
                    'subtype': parts[5].strip(),
                    'links': []
                }

                # 解析关联点
                if len(parts) > 6:
                    links_str = parts[6].strip()
                    if links_str:
                        for link in links_str.split(';'):
                            link = link.strip()
                            if link:
                                link_parts = [p.strip() for p in link.split(',')]
                                if len(link_parts) >= 3:
                                    minutiae['links'].append({
                                        'id': int(link_parts[0]),
                                        'x': int(link_parts[1]),
                                        'y': int(link_parts[2])
                                    })
                                else:
                                    print(f"警告：第 {line_num} 行关联点格式错误 '{link}'")
                minutiae_data.append(minutiae)

            except Exception as e:
                print(f"错误：第 {line_num} 行解析失败（{e}），已跳过")
    return minutiae_data


def plot_minutiae(minutiae_data, img_size=(256, 416)):
    """可视化细节点（同上，略）"""
    pass


if __name__ == "__main__":
    file_path = "26_3_features.min"  # 替换为实际路径
    minutiae_data = read_min_file(file_path)
    print(f"成功解析 {len(minutiae_data)} 个细节点")
    plot_minutiae(minutiae_data)
