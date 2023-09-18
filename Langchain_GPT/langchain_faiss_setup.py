import os
import numpy as np
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
metadata_path = os.path.join(script_dir, 'metadata.json')
api_key_file_path = os.path.join(script_dir, 'api_key.txt')

def initialize_langchain_and_faiss():
    # 初始化OpenAI嵌入
    openai_embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")

    # 初始化FAISS索引
    faiss_index = FAISS(vector_dim=openai_embeddings.vector_dim)

    # 检查是否已经有现有的FAISS索引和嵌入
    if os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
        faiss_index = faiss.read_index(faiss_index_path)
        embeddings = np.load(embeddings_path)
    else:
        # 在这里，您可以添加代码来创建新的FAISS索引和嵌入
        pass

    return openai_embeddings, faiss_index, embeddings
#加载GPT-API-Key的函数
def load_api_key():
    try:
        with open(api_key_file_path, "r") as key_file:
            api_key = key_file.read().strip()
    except FileNotFoundError:
        api_key = input("请输入您的OpenAI API密钥：")
    return api_key
