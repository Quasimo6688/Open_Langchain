import pdfplumber
import os
import networkx as nx
import json
from PIL import Image
import io

# 设置目标文件夹路径
folder_path = '/mnt/data/Embedding_Files'

# 初始化网络图，用于存储知识图谱
G = nx.DiGraph()

# 遍历目标文件夹中的所有PDF文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(folder_path, file_name)

        # 使用pdfplumber打开PDF文件
        with pdfplumber.open(pdf_path) as pdf:
            # 初始化变量以存储图片信息
            image_pages = []

            # 遍历PDF的每一页
            for i, page in enumerate(pdf.pages):
                # 提取文本
                text = page.extract_text()
                if text:
                    # 如果文本超过120字，则进行分割
                    if len(text) > 150:
                        start = 0
                        while start < len(text):
                            end = min(start + 150, len(text))
                            text_block = text[start:end]
                            block_node = f"{file_name}_文本_页{i + 1}_块{start // 150 + 1}"
                            G.add_node(block_node, 类型="文本块", 文件名=file_name, 页码=i + 1, 内容=text_block)
                            start = end - 20  # 设置20字重叠
                    else:
                        # 如果文本不超过150字，则直接添加整页文本
                        G.add_node(f"{file_name}_文本_页{i + 1}", 类型="整页文本", 文件名=file_name, 页码=i + 1, 内容=text)

                # 提取并保存图片
                image_folder_path = os.path.join('Embedding_Files', 'Pictures')
                os.makedirs(image_folder_path, exist_ok=True)
                for image_index, img in enumerate(page.images):
                    image_bytes = page.extract_image(img["x0"], img["top"], img["x1"], img["bottom"])["image"]
                    image_path = os.path.join(image_folder_path, f'{file_name}_页{i + 1}_图片{image_index + 1}.png')
                    Image.ope
