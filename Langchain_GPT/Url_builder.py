import os
import json

def generate_url_pic_map(directory, output_file, base_url):
    print("开始生成 URL_Pic_map.json 文件...")

    # 创建空字典用于存储文件名和URL的映射
    url_pic_map = {}

    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # 假设目标文件是PNG格式
            # 生成每个文件的URL
            url = os.path.join(base_url, filename)
            url_pic_map[filename] = url

    # 将字典写入JSON文件
    with open(os.path.join(directory, output_file), 'w', encoding='utf-8') as file:
        json.dump(url_pic_map, file, indent=4, ensure_ascii=False)
    print(f"已成功生成 {output_file} 在 {directory}")

# 使用示例
directory = '/home/project/pyxl-data/xuan-faiss/Embedding_Files/Pictures'  # 目标文件夹路径
output_file = 'URL_Pic_map.json'  # 输出文件名
base_url = 'http://example.com/'  # 基本URL，您需要根据实际情况进行更改
generate_url_pic_map(directory, output_file, base_url)
