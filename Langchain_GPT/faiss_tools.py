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
import pdfplumber
from tqdm import tqdm
import re
import hashlib
import io
from io import BytesIO
from PIL import Image


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
    清洗整个文本块，包括标点符号规范化、去除特殊字符、段落合并和去除多余换行。
    """
    # 规范化标点符号（将全角标点转换为半角）
    text_block = re.sub(r'[\u3000\u3001\u3002\uff0c\uff0e\uff1b\uff1f\uff01\uff1a\u201c\u201d\u2018\u2019]',
                        lambda m: {u'\u3000': ' ', u'\u3001': ',', u'\u3002': '.', u'\uff0c': ',', u'\uff0e': '.',
                                   u'\uff1b': ';', u'\uff1f': '?', u'\uff01': '!', u'\uff1a': ':',
                                   u'\u201c': '"', u'\u201d': '"', u'\u2018': "'", u'\u2019': "'"}.get(m.group()),
                        text_block)

    # 去除特殊字符（例如HTML标签和连续的点号）
    text_block = re.sub(r'<[^>]+>', '', text_block)  # 假设HTML标签是不需要的
    text_block = re.sub(r'\.{2,}', ' ', text_block)  # 去除连续的点号

    # 这里假设如果一个段落（非空行）后面跟着的是另一个段落（非空行），它们应该合并
    text_block = re.sub(r'(?<=\S)\n(?=\S)', ' ', text_block)

    # 清除额外的空白字符，包括空格和换行
    text_block = ' '.join(text_block.split())

    return text_block


def extract_and_save_images(page, page_index, file_name, target_folder_path, picture_map, image_hashes):
    image_folder_path = os.path.join(target_folder_path, 'Embedding_Files', 'Pictures')
    os.makedirs(image_folder_path, exist_ok=True)
    total_images = 0
    filtered_images = 0

    def compute_image_hash(image):
        hash_md5 = hashlib.md5()
        hash_md5.update(image.tobytes())
        return hash_md5.hexdigest()

    def correct_bbox(bbox, page_bbox):
        x0, top, x1, bottom = bbox
        px0, ptop, px1, pbottom = page_bbox
        x0 = max(x0, px0)
        top = max(top, ptop)
        x1 = min(x1, px1)
        bottom = min(bottom, pbottom)
        return (x0, top, x1, bottom)

    logger.info(f"Starting image extraction for page {page_index + 1} of {file_name}.")

    for image_index, img in enumerate(page.images):
        bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
        corrected_bbox = correct_bbox(bbox, page.bbox)
        cropped_page = page.crop(corrected_bbox)

        if cropped_page.width == 0 or cropped_page.height == 0:
            continue

        total_images += 1

        try:
            pil_image = cropped_page.to_image().original
            image_hash = compute_image_hash(pil_image)

            if image_hash not in image_hashes:
                image_hashes.add(image_hash)
                image_file_name = f'{file_name}_page_{page_index + 1}_image_{image_index + 1}.png'
                image_file_path = os.path.join(image_folder_path, image_file_name)
                pil_image.save(image_file_path, format='PNG')

                image_node_name = f"{file_name}_page_{page_index + 1}_image_{image_index + 1}"
                picture_map[image_node_name] = {
                    'page': page_index + 1,
                    'image_index': image_index + 1,
                    'image_path': image_file_path
                }
            else:
                filtered_images += 1
        except Exception as e:
            logger.error(f"Error processing or saving image: {e}")

    logger.info(f"Image extraction completed for page {page_index + 1} of {file_name}.")
    logger.info(f"Total images extracted: {total_images - filtered_images}")
    logger.info(f"Total duplicate images filtered: {filtered_images}")

    pictures_map_file_path = os.path.join(target_folder_path, 'Embedding_Files', 'Pictures_map.json')
    try:
        with open(pictures_map_file_path, 'w', encoding='utf-8') as file:
            json.dump(picture_map, file, ensure_ascii=False, indent=4)
        logger.info(f"Picture map JSON file saved at '{pictures_map_file_path}'.")
    except Exception as e:
        logger.error(f"Error saving picture map JSON file at '{pictures_map_file_path}': {e}")

def split_text_into_blocks(text_content, page_index, file_name, max_length=150, overlap=20):
    logger.info(f"开始分割文件 '{file_name}' 中的文本")
    combined_blocks = []
    block = ''
    block_index = 0  # 初始化文本块索引

    # 中文常用标点符号，用于辅助判断分割点
    punctuation = '。！？；，'

    for char in text_content:
        block += char
        if len(block) >= max_length or (char in punctuation and len(block) >= max_length - overlap):
            block_index += 1
            combined_blocks.append({
                "content": block.strip(),
                "page": page_index + 1,
                "block_index": block_index,
                "identifier": f"{file_name}_page_{page_index + 1}_block_{block_index}"
            })
            block = block[-overlap:]

    # 添加最后一个块（如果有）
    if block:
        block_index += 1
        combined_blocks.append({
            "content": block.strip(),
            "page": page_index + 1,
            "block_index": block_index,
            "identifier": f"{file_name}_page_{page_index + 1}_block_{block_index}"
        })

    logger.info(f"文本分割完成，共分割成 {len(combined_blocks)} 个块")
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
    vectors = []
    logger.info(f"开始向量化文件 '{filename}'")

    for i, block in enumerate(tqdm(blocks, desc=f"向量化进度", unit="block")):
        vector = vectorize_block(block)
        if vector is not None:
            vectors.append(vector)

    logger.info(f"{filename} 向量化完成，共处理 {len(blocks)} 个文本块")
    return np.array(vectors)

def save_document_collection(blocks, filename):
    """
    将文本块内容及其相关信息保存为JSON文件。
    """
    document_collection = {
        i: {
            'content': block['content'],
            'page': block['page'],
            'identifier': block['identifier']
        }
        for i, block in enumerate(blocks)
    }
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

    # 初始化图片映射字典和图片哈希集合
    picture_map = {}
    image_hashes = set()

    # 使用新函数获取PDF文件列表
    pdf_files = find_pdf_files(folder_path)

    # 统一处理所有PDF文件
    all_blocks = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        logger.info(f"正在处理文件：{filename}")

        # 提取和清洗标题（停用）
        #titles = extract_titles_from_pdf(pdf_path)
        #cleaned_titles = clean_titles(titles, filter_keywords)

        # 提取和处理文本
        with open(pdf_path, 'rb') as file:
            text_content = extract_text(file)
            cleaned_text = clean_text_block(text_content)
            blocks = split_text_into_blocks(cleaned_text, 0, filename)  # 假设整个文档视为一页
            all_blocks.extend(blocks)

        # 处理图片提取
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                extract_and_save_images(page, page_index, filename, script_dir, picture_map, image_hashes)

    # 向量化所有文本块
    all_vectors = convert_blocks_to_vectors(all_blocks, "combined")

    # 保存集合文本和向量
    document_collection_filename = "combined_collection.json"
    vector_filename = "combined_vectors.npy"
    save_document_collection(all_blocks, document_collection_filename)
    save_vectors(all_vectors, vector_filename)

    # 创建并保存FAISS索引
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
