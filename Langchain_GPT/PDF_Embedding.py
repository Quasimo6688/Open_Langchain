import os
import time
from pdfminer.high_level import extract_text
from tqdm import tqdm
import logging

# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 判断是否为标题的函数
def is_title(line, prev_line, next_line):
    return prev_line.strip() == '' and line.strip() != '' and next_line.strip() == ''

# 从PDF文件中提取标题
def extract_titles_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    lines = text.split('\n')
    titles = []

    for i in range(1, len(lines) - 1):
        if is_title(lines[i], lines[i - 1], lines[i + 1]):
            titles.append(lines[i])

    return titles

# 清洗标题列表，移除包含特定关键词的项
def clean_titles(titles, filter_keywords=None):
    if filter_keywords is None:
        filter_keywords = []

    cleaned_titles = []
    for title in titles:
        if not any(keyword in title for keyword in filter_keywords):
            cleaned_titles.append(title)
        else:
            logger.info(f"过滤掉的标题: {title}")
    return cleaned_titles

# 从文件夹中提取所有PDF文件的标题并进行清洗
def extract_and_clean_titles(folder_path, filter_keywords=None):
    titles_in_files = {}
    for filename in tqdm(os.listdir(folder_path), desc="提取标题"):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            titles = extract_titles_from_pdf(pdf_path)
            cleaned_titles = clean_titles(titles, filter_keywords)
            titles_in_files[filename] = cleaned_titles
            logger.info(f"文件：{filename}，标题提取完成")
    return titles_in_files

# 清洗文本块
def clean_text_block(text_block):
    # 替换多个连续空格为单个空格
    text_block = ' '.join(text_block.split())
    # 移除字符串两端的空白字符
    text_block = text_block.strip()
    return text_block

# 分割文本块的函数
def split_text_into_blocks(titles, text_content, chunk_size=500):
    combined_blocks = []
    for index, title in tqdm(enumerate(titles), total=len(titles), desc="分割文本"):
        start_idx = text_content.find(title)
        end_idx = text_content.find(titles[index + 1], start_idx + 1) if index + 1 < len(titles) else len(text_content)
        content_block = text_content[start_idx:end_idx].strip()

        if len(content_block) <= chunk_size:
            combined_blocks.append({'title': title, 'content': clean_text_block(content_block)})
        else:
            current_block = title
            for word in content_block[len(title):].split():
                if len(current_block) + len(word) + 1 <= chunk_size:
                    current_block += ' ' + word
                else:
                    combined_blocks.append({'title': title, 'content': clean_text_block(current_block.strip())})
                    current_block = title + ' ' + word
            if current_block.strip() != title:
                combined_blocks.append({'title': title, 'content': clean_text_block(current_block.strip())})
    return combined_blocks

# 主函数
def main():
    folder_path = 'uploaded_files'  # 假设这是您的文件夹路径
    filter_keywords = ['或者', 'PDF', '示例问题', '例子', '解释', '如图']  # 标题过滤关键词
    all_titles = extract_and_clean_titles(folder_path, filter_keywords)

    start_time = time.time()
    for filename, titles in all_titles.items():
        pdf_path = os.path.join(folder_path, filename)
        text_content = extract_text(pdf_path)
        blocks = split_text_into_blocks(titles, text_content)
        logger.info(f"文件：{filename}，分割完成")
        for block in blocks:
            logger.info(f"标题：{block['title']}, 内容：{block['content'][:50]}...")  # 记录每个块的标题和部分内容

    end_time = time.time()
    logger.info(f"文本分割和记录完成，耗时：{end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main()
