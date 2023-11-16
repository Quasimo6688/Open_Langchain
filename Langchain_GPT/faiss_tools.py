import os
import time
from pdfminer.high_level import extract_text
from tqdm import tqdm
import logging
import numpy as np
import zhipuai  # 确保已安装并导入zhipuai库
import json


# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

embedding_files_dir = os.path.join(os.getcwd(), 'Embedding_Files')

# 设置zhipuai API密钥
zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"

def is_title(line, prev_line, next_line):
    return prev_line.strip() == '' and line.strip() != '' and next_line.strip() == ''

# 从文件夹中提取所有PDF文件的标题
def extract_titles_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    lines = text.split('\n')
    titles = []
    title_counts = {}  # 用于跟踪每个标题出现的次数

    for i in range(1, len(lines) - 1):
        if is_title(lines[i], lines[i - 1], lines[i + 1]):
            title = lines[i]
            if title in title_counts:
                # 如果标题已经存在，增加计数并添加尾号
                title_counts[title] += 1
                title_with_suffix = f"{title}_{title_counts[title]}"
                titles.append(title_with_suffix)
            else:
                # 如果是首次出现的标题，添加到字典并设置计数为1
                title_counts[title] = 1
                titles.append(title)

    logger.info(f"在文件 {os.path.basename(pdf_path)} 中找到 {len(titles)} 个标题")
    return titles


def clean_titles(titles, filter_keywords=None):
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
    text_block = ' '.join(text_block.split())
    text_block = text_block.strip()
    return text_block



def split_text_into_blocks(text_content, max_length=512, overlap=100):
    logger.info(f"开始进行文件分割")
    combined_blocks = []
    start_idx = 0
    total_chars = 0  # 用于记录总字符数

    while start_idx < len(text_content):
        # 确定当前块的结束位置
        end_idx = min(start_idx + max_length, len(text_content))

        # 提取当前块的内容
        current_block = text_content[start_idx:end_idx].strip()
        combined_blocks.append({"content": current_block})
        logger.info(f"完成了第{len(combined_blocks)}个块")

        # 更新总字符数
        total_chars += len(current_block)

        # 更新下一个块的开始位置
        start_idx = end_idx - overlap

        # 确保开始索引不会变成负数
        if start_idx < 0:
            start_idx = 0

        # 确保每次迭代后start_idx都有所增加，避免无限循环
        if end_idx - start_idx <= overlap:
            start_idx = end_idx

    logger.info(f"文本分割成 {len(combined_blocks)} 个块，总字符数为 {total_chars}")
    return combined_blocks




def vectorize_block(block, result_list, index, max_retries=3):
    retries = 0
    while retries <= max_retries:
        try:
            response = zhipuai.model_api.invoke(
                model="text_embedding",
                prompt=block['content']
            )
            if response and response.get('code') == 200 and response.get('success'):
                vector = response['data']['embedding']
                result_list[index] = vector
                logger.info(f"向量化成功: 文本块索引 {index}")
                return True
            else:
                logger.warning(f"向量化失败，重试中...: 文本块索引 {index}, 重试次数 {retries}")
                retries += 1
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"向量化过程中发生异常: {e}")
            retries += 1
            time.sleep(0.5)

    logger.error(f"向量化最终失败: 文本块索引 {index}")
    result_list[index] = None
    return False


def convert_blocks_to_vectors(blocks, filename):
    vectors = [None] * len(blocks)
    logger.info(f"开始向量化文件：{filename}")

    with tqdm(total=len(blocks), desc=f"向量化进度 - {filename}") as progress_bar:
        for i, block in enumerate(blocks):
            vectorize_block(block, vectors, i)
            progress_bar.update(1)  # 在这里更新进度条

    # 过滤掉向量化失败的块
    vectors = [v for v in vectors if v is not None]

    logger.info(f"{filename}向量化完成")
    return vectors

def save_document_collection(blocks, filename):
    # 创建一个包含所有文本块内容的字典
    document_collection = {i: block['content'] for i, block in enumerate(blocks)}

    # 保存为 JSON 文件
    collection_path = os.path.join(embedding_files_dir, f"{filename}_collection.json")
    with open(collection_path, 'w', encoding='utf-8') as file:
        json.dump(document_collection, file, ensure_ascii=False, indent=4)
    logger.info(f"{filename} 的文档集合保存完成")

def save_vectors(vectors, filename):
    if not os.path.exists(embedding_files_dir):
        os.makedirs(embedding_files_dir)

    full_path = os.path.join(embedding_files_dir, filename)
    np.save(full_path, vectors)
    logger.info(f"{filename}保存完成")


def main():
    folder_path = 'uploaded_files'
    filter_keywords = ['或者', 'PDF', '示例问题', '例子', '解释', '如图']

    # 提取并清洗所有文件的标题
    all_titles = {}
    all_cleaned_texts = {}  # 用于存储清洗后的完整文本
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            titles = extract_titles_from_pdf(pdf_path)
            cleaned_titles = clean_titles(titles, filter_keywords)
            all_titles[filename] = cleaned_titles

            # 读取并清洗整个文档的文本
            with open(pdf_path, 'rb') as file:
                text_content = extract_text(file)
                cleaned_text = clean_text_block(text_content)
                all_cleaned_texts[filename] = cleaned_text

            # 添加的日志记录语句
            logger.info(f"文件 {filename} 的文本提取和清洗完成")

    # 分割所有文件的文本
    all_blocks = {}
    for filename in all_titles.keys():
        # 使用清洗后的文本进行分割
        cleaned_text = all_cleaned_texts[filename]

        # 在这里添加日志记录，显示待分割文件的整体字数量
        logger.info(f"文件 {filename} 的总字符数为：{len(cleaned_text)}")

        blocks = split_text_into_blocks(cleaned_text, max_length=512, overlap=100)
        logger.info(f"{filename} 分割完成，总文字数：{sum(len(block['content']) for block in blocks)}")
        all_blocks[filename] = blocks

    # 向量化处理
    for filename, blocks in all_blocks.items():
        file_vectors = convert_blocks_to_vectors(blocks, filename)
        save_vectors(file_vectors, f"{filename}_vectors.npy")
        save_document_collection(blocks, filename)  # 保存文档集合
        logger.info(f"{filename}保存完成")

if __name__ == "__main__":
    main()

