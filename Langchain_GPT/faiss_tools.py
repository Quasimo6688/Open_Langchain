import os
import logging
import numpy as np
import faiss
import zhipuai
import json
import threading
import jieba
from pdfminer.high_level import extract_text
from tqdm import tqdm
import time
import concurrent.futures

# 初始化日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置zhipuai API密钥
zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"

# 全局变量定义
script_dir = os.path.dirname(os.path.abspath(__file__))

def find_pdf_files(folder_path):
    logger.info(f"开始检查路径")
    """
    在指定文件夹中查找所有PDF文件。
    """
    pdf_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            pdf_files.append(pdf_path)

    logger.info(f"在 {folder_path} 中找到 {len(pdf_files)} 个PDF文件")
    return pdf_files

def is_title(line, prev_line, next_line):
    """
    判断一行文本是否是标题。
    """
    return prev_line.strip() == '' and line.strip() != '' and next_line.strip() == ''

def extract_titles_from_pdf(pdf_path):
    """
    从PDF文件中提取标题。
    """
    text = extract_text(pdf_path)
    lines = text.split('\n')
    titles = []
    title_counts = {}

    for i in range(1, len(lines) - 1):
        if is_title(lines[i], lines[i - 1], lines[i + 1]):
            title = lines[i]
            if title in title_counts:
                title_counts[title] += 1
                title_with_suffix = f"{title}_{title_counts[title]}"
                titles.append(title_with_suffix)
            else:
                title_counts[title] = 1
                titles.append(title)

    logger.info(f"在文件 {os.path.basename(pdf_path)} 中找到 {len(titles)} 个标题")
    return titles

def clean_titles(titles, filter_keywords=None):
    """
    清洗提取出的标题。
    """
    if filter_keywords is None:
        filter_keywords = []

    cleaned_titles = []
    for title in titles:
        if not any(keyword in title for keyword in filter_keywords):
            cleaned_titles.append(title)
        else:
            logger.info(f"过滤掉的标题: {title}")
    logger.info(f"过滤后剩余标题数量: {len(cleaned_titles)}")
    return cleaned_titles

def clean_text_block(text_block):
    """
    清洗整个文本块。
    """
    text_block = ' '.join(text_block.split())
    text_block = text_block.strip()
    return text_block

def split_text_into_blocks(text_content, max_length=120, overlap=20):
    """
    使用 jieba 分词库和标点符号将中文文本分割成更小的块，同时实现块之间的重叠。
    """
    logger.info(f"开始进行文件分割")
    combined_blocks = []
    block = ''

    # 中文常用标点符号，用于辅助判断分割点
    punctuation = '。！？；，'

    for char in text_content:
        block += char
        if len(block) >= max_length or (char in punctuation and len(block) >= max_length - overlap):
            combined_blocks.append({"content": block.strip()})
            block = block[-overlap:]  # 保持重史

    # 添加最后一个块（如果有）
    if block:
        combined_blocks.append({"content": block.strip()})

    logger.info(f"文本分割成 {len(combined_blocks)} 个块")
    return combined_blocks

def vectorize_block(block):
    """
    使用zhipuai库将文本块转换成向量，并在每次请求后增加 0.2 秒的延迟。
    """
    try:
        response = zhipuai.model_api.invoke(
            model="text_embedding",
            prompt=block['content']
        )
        time.sleep(0.1)  # 请求后延时
        if response and response.get('code') == 200 and response.get('success'):
            return response['data']['embedding']
        else:
            logger.warning("向量化失败")
    except Exception as e:
        logger.error(f"向量化过程中发生异常: {e}")
    return None

def convert_blocks_to_vectors(blocks, filename):
    """
    将文本块转换为向量，并保存结果。
    使用单线程顺序处理每个文本块。
    """
    vectors = []
    logger.info(f"开始向量化文件：{filename}")

    for i, block in enumerate(blocks):
        vector = vectorize_block(block)
        if vector is not None:
            vectors.append(vector)
        logger.info(f"向量化进度: {i + 1}/{len(blocks)}")

    logger.info(f"{filename}向量化完成")
    return np.array(vectors)

def save_document_collection(blocks, filename):
    """
    将文本块内容保存为JSON文件。
    """
    document_collection = {i: block['content'] for i, block in enumerate(blocks)}
    collection_path = os.path.join(script_dir, 'Embedding_Files', f"{filename}_collection.json")
    with open(collection_path, 'w', encoding='utf-8') as file:
        json.dump(document_collection, file, ensure_ascii=False, indent=4)
    logger.info(f"{filename} 的文档集合保存完成")

def save_vectors(vectors, filename):
    """
    将向量保存为NumPy文件。
    """
    embedding_dir = os.path.join(script_dir, 'Embedding_Files')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    full_path = os.path.join(embedding_dir, f"{filename}_vectors.npy")
    np.save(full_path, vectors)
    logger.info(f"{filename}保存完成")


def create_and_save_faiss_index(vectors, index_path):
    """
    创建并保存FAISS索引。
    :param vectors: NumPy数组形式的向量
    :param index_path: 索引保存路径
    """
    logger.info("开始创建FAISS索引")
    # 获取向量维度
    d = vectors.shape[1]
    # 创建使用L2距离的Flat索引
    index = faiss.IndexFlatL2(d)
    # 添加向量到索引
    index.add(vectors)
    # 保存索引到磁盘
    faiss.write_index(index, index_path)
    logger.info(f"FAISS索引已保存到 {index_path}")


def main():
    folder_path = 'uploaded_files'
    filter_keywords = ['过滤词1', '过滤词2']

    # 0. 使用新函数获取PDF文件列表
    pdf_files = find_pdf_files(folder_path)

    # 1. 统一处理所有PDF文件
    all_blocks = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"正在处理文件：{filename}")

        titles = extract_titles_from_pdf(pdf_path)
        cleaned_titles = clean_titles(titles, filter_keywords)

        with open(pdf_path, 'rb') as file:
            text_content = extract_text(file)
            cleaned_text = clean_text_block(text_content)
            blocks = split_text_into_blocks(cleaned_text)
            all_blocks.extend(blocks)

    # 2. 向量化所有文本块
    all_vectors = convert_blocks_to_vectors(all_blocks, "combined")

    # 3. 保存集合文本和向量
    document_collection_filename = "combined_collection.json"
    vector_filename = "combined_vectors.npy"
    save_document_collection(all_blocks, document_collection_filename)
    save_vectors(all_vectors, vector_filename)

    # 4. 创建并保存FAISS索引
    faiss_index_path = os.path.join(script_dir, 'Embedding_Files', 'faiss_glm.index')
    create_and_save_faiss_index(all_vectors, faiss_index_path)

    # 构建文件路径
    file_paths = {
        'document_collection': os.path.join(script_dir, 'Embedding_Files', document_collection_filename),
        'vectors': os.path.join(script_dir, 'Embedding_Files', vector_filename),
        'faiss_index': faiss_index_path
    }

    logger.info("所有处理步骤完成。")
    return file_paths

if __name__ == "__main__":
    main()
