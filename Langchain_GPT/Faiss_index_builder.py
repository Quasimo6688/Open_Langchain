import os
import numpy as np
import faiss
from tqdm import tqdm
import logging

# 设置日志记录的基本配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

embedding_dir = 'Embedding_Files'
vectors = []

logging.info("开始加载向量文件")

# 遍历 Embedding_Files 文件夹，加载所有向量文件
for file in tqdm(os.listdir(embedding_dir), desc="加载向量"):
    if file.endswith('.npy'):
        file_path = os.path.join(embedding_dir, file)
        vectors.append(np.load(file_path))

logging.info("向量文件加载完成，开始创建索引")

# 将所有向量合并成一个大的 NumPy 数组
all_vectors = np.concatenate(vectors, axis=0)

dimension = all_vectors.shape[1]  # 向量的维度
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离

# 一次性添加所有向量到索引
index.add(all_vectors)

logging.info("索引创建完成，开始保存索引文件")

# 保存索引到文件
index_save_path = os.path.join(os.getcwd(), 'Embedding_Files/faiss_glm.index')
faiss.write_index(index, index_save_path)

logging.info("索引文件保存成功")
