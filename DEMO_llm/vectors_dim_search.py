import numpy as np
import os

# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建嵌入向量文件的完整路径
embeddings_file = os.path.join(script_dir, 'Embedding_Files', '航空知识手册全集_下册.pdf_vectors.npy')

# 加载嵌入向量
embeddings = np.load(embeddings_file)

# 获取向量维度
vector_dim = embeddings.shape[1]
print(f"向量维度是: {vector_dim}")
