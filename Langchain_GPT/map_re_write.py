import json
import os

def update_image_paths(url_map_file, pictures_map_file, base_dir):
    print("开始更新图片路径...")

    # 构建完整的文件路径
    url_map_path = os.path.join(base_dir, url_map_file)
    pictures_map_path = os.path.join(base_dir, pictures_map_file)

    # 加载URL_Pictures_map文件
    with open(url_map_path, 'r', encoding='utf-8') as file:
        url_map = json.load(file)
    print(f"已加载 {url_map_file}")

    # 加载Pictures_map文件
    with open(pictures_map_path, 'r', encoding='utf-8') as file:
        pictures_map = json.load(file)
    print(f"已加载 {pictures_map_file}")

    # 更新Pictures_map中的image_path
    updated_count = 0
    for key, url in url_map.items():
        # 移除键名称中的.png后缀
        modified_key = key.rsplit('.png', 1)[0]
        if modified_key in pictures_map:
            pictures_map[modified_key]['image_path'] = url
            updated_count += 1
            print(f"已更新 {modified_key} 的 image_path")

    # 保存更新后的Pictures_map文件
    with open(pictures_map_path, 'w', encoding='utf-8') as file:
        json.dump(pictures_map, file, indent=4, ensure_ascii=False)
    print(f"已保存更新后的 {pictures_map_file}，共更新 {updated_count} 条目")

# 使用示例
base_dir = 'Embedding_Files'  # 程序所在位置的Embedding_Files文件夹
url_map_file = 'URL_Pic_map.json'  # URL_Pictures_map文件名
pictures_map_file = 'Pictures_map.json'  # Pictures_map文件名
update_image_paths(url_map_file, pictures_map_file, base_dir)
