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
    logger.info("开始清洗文本块")
    # 规范化标点符号（将全角标点转换为半角）
    text_block = re.sub(r'[\u3000\u3001\u3002\uff0c\uff0e\uff1b\uff1f\uff01\uff1a\u201c\u201d\u2018\u2019]',
                        lambda m: {u'\u3000': ' ', u'\u3001': ',', u'\u3002': '.', u'\uff0c': ',', u'\uff0e': '.',
                                   u'\uff1b': ';', u'\uff1f': '?', u'\uff01': '!', u'\uff1a': ':',
                                   u'\u201c': '"', u'\u201d': '"', u'\u2018': "'", u'\u2019': "'"}.get(m.group()),
                        text_block)

    # 去除特殊字符（例如HTML标签和连续的点号）
    text_block = re.sub(r'<[^>]+>', '', text_block)  # 假设HTML标签是不需要的
    text_block = re.sub(r'\.{2,}', ' ', text_block)  # 去除连续的点号

    # 合并段落
    text_block = re.sub(r'(?<=\S)\n(?=\S)', ' ', text_block)

    # 清除额外的空白字符
    text_block = ' '.join(text_block.split())

    # 添加日志记录点
    logger.info("文本块清洗完成")
    # 打印清洗后的文本块内容
    return text_block


def extract_and_save_images(page, page_index, file_name, target_folder_path, picture_map, image_hashes):
    logger.info(f"开始提取第 {page_index + 1} 页的图片，文件：{file_name}")
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
        logger.info(f"图片提取完成，第 {page_index + 1} 页，文件：{file_name}")

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
        logger.info(f"图片保存完成，第 {page_index + 1} 页，文件：{file_name}")
    except Exception as e:
        logger.error(f"Error saving picture map JSON file at '{pictures_map_file_path}': {e}")


def split_text_into_blocks(text_content, global_page_number, file_name, pdf_path, max_length=550, overlap=50):
    logger.info(f"开始分割文本块，文件：{file_name}, 页码：{global_page_number}")
    blocks = []
    start = 0
    text_length = len(text_content)
    block_index = 0  # 初始化文本块索引

    while start < text_length:
        # 计算块的结束位置
        end = min(start + max_length, text_length)
        block = text_content[start:end].strip()

        # 只有在块非空时才处理
        if block:
            block_index += 1
            # 将页码添加到文本块的开头，并使用特定符号标记
            formatted_block = f"[{file_name}] [{global_page_number}] {block} [{pdf_path}]"
            blocks.append(formatted_block)

        # 更新下一个块的起始位置
        start = end if end == text_length else end - overlap

        # 添加日志记录点
        logger.info(f"处理完成一个文本块，文件：{file_name}, 块索引：{block_index}, 页码：{global_page_number}")

    logger.info(f"文本块分割完成，文件：{file_name}, 页码：{global_page_number}")
    return blocks





def process_pdf_file(pdf_path, global_page_number, picture_map, image_hashes):
    all_blocks = []
    file_name = os.path.basename(pdf_path)
    logger.info(f"开始处理文件: {file_name}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"打开文件: {file_name}")
            for page in pdf.pages:
                global_page_number += 1
                logger.info(f"处理第 {global_page_number} 页")

                # 文本提取
                page_text = page.extract_text()
                if page_text:
                    logger.info(f"第 {global_page_number} 页提取文本成功")
                    cleaned_text = clean_text_block(page_text)
                    logger.info(f"第 {global_page_number} 页文本清洗完成")
                    blocks = split_text_into_blocks(cleaned_text, global_page_number, file_name, pdf_path)
                    logger.info(f"第 {global_page_number} 页文本分割完成")
                    all_blocks.extend(blocks)
                else:
                    logger.warning(f"第 {global_page_number} 页无可提取文本")

                # 图片提取
                #extract_and_save_images(page, global_page_number, file_name, script_dir, picture_map, image_hashes)

    except Exception as e:
        logger.error(f"处理文件 {file_name} 时出错: {e}", exc_info=True)

    return all_blocks, global_page_number



def vectorize_block(block, retries=30):
    """
    使用zhipuai库将文本块转换成向量，并在每次请求后增加 0.2 秒的延迟。
    在失败时重试最多 retries 次。
    """
    attempt = 0
    while attempt < retries:
        try:
            response = zhipuai.model_api.invoke(
                model="text_embedding",
                prompt=block
            )
            time.sleep(0.1)  # 请求后延时
            if response and response.get('code') == 200 and response.get('success'):
                return response['data']['embedding']
            else:
                logger.warning(f"第 {attempt + 1} 次向量化尝试失败")
                attempt += 1
                time.sleep(3)  # 在重试之前增加一些延时
        except Exception as e:
            logger.error(f"向量化过程中发生异常: {e}")
            attempt += 1
            time.sleep(3)  # 在重试之前增加一些延时
    logger.warning(f"文本块向量化重试 {retries} 次后仍然失败")
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
    # 将每个文本块转换为包含文件名、页码、内容和文件路径的字典
    blocks_dict = {}
    for i, block in enumerate(blocks):
        # 分离文件名、页码、文本内容和文件路径
        match = re.search(r'^\[(.*?)\] \[(\d+)\] (.*?) \[(.*?)\]$', block, re.DOTALL)
        if match:
            file_name = match.group(1)
            page_number = int(match.group(2))
            text_content = match.group(3).strip()
            pdf_path = match.group(4)
            blocks_dict[i] = {
                "file_name": file_name,
                "page_number": page_number,
                "content": text_content,
                "pdf_path": pdf_path
            }
        else:
            logger.warning(f"无法从文本块中提取完整信息：{block}")

    # 保存字典到 JSON 文件
    collection_path = os.path.join(script_dir, 'Embedding_Files', f"{filename}.json")
    with open(collection_path, 'w', encoding='utf-8') as file:
        json.dump(blocks_dict, file, ensure_ascii=False, indent=4)
    logger.info(f"{filename} 的文档集合保存完成")






def save_vectors(vectors, filename):
    """
    将向量保存为NumPy文件。
    """
    embedding_dir = os.path.join(script_dir, 'Embedding_Files')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    full_path = os.path.join(embedding_dir, f"{filename}")
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
    if vectors.size > 0:
        index.add(vectors)
        logger.info("向量已成功添加到FAISS索引")
    else:
        logger.error("无法添加向量到索引，因为向量数组为空")
    # 保存索引到磁盘
    faiss.write_index(index, index_path)
    logger.info(f"FAISS索引已保存到 {index_path}")



def main():
    folder_path = 'uploaded_files'
    logger.info("程序开始运行")

    # 初始化图片映射字典
    picture_map = {}
    image_hashes = set()

    # 获取PDF文件列表
    pdf_files = find_pdf_files(folder_path)

    # 初始化文本提取相关变量
    all_blocks_across_files = []
    global_page_number = 0  # 全局页码计数器
    logger.info("开始文本提取、清洗、分割过程")

    # 遍历PDF文件进行文本提取、清洗、分割，图片提取分割、哈希过滤
    for pdf_path in pdf_files:
        logger.info(f"处理文件：{os.path.basename(pdf_path)}")
        blocks, global_page_number = process_pdf_file(pdf_path, global_page_number, picture_map, image_hashes)
        all_blocks_across_files.extend(blocks)
    logger.info("文本提取、清洗、分割过程完成")

    # 保存处理过的文本块到集合文档
    logger.info("开始保存处理过的文本块")
    save_document_collection(all_blocks_across_files, "combined_collection")
    logger.info("文本块保存完成")

    # 向量化之前的询问步骤
    proceed = input("是否继续进行向量化？(y/n): ")
    if proceed.lower() != 'y':
        logger.info("程序终止。")
        return

    # 向量化所有文本块
    logger.info("开始向量化文本块")
    all_vectors = convert_blocks_to_vectors(all_blocks_across_files, "combined")
    logger.info("文本块向量化完成")

    # 保存向量
    logger.info("开始保存向量")
    vector_filename = "combined_vectors"
    save_vectors(all_vectors, vector_filename)
    logger.info("向量保存完成")

    # 创建并保存FAISS索引
    logger.info("开始创建并保存FAISS索引")
    faiss_index_path = os.path.join(script_dir, 'Embedding_Files', 'Files_faiss_glm.index')
    create_and_save_faiss_index(all_vectors, faiss_index_path)
    logger.info("FAISS索引创建并保存完成")

    # 构建文件路径
    file_paths = {
        'document_collection': os.path.join(script_dir, 'Embedding_Files', "Files_combined_collection.json"),
        'vectors': os.path.join(script_dir, 'Embedding_Files', vector_filename+".npy"),
        'faiss_index': faiss_index_path
    }

    logger.info("所有处理步骤完成。")
    return file_paths

if __name__ == "__main__":
    main()
