import pdfplumber
import os
import jieba
from gensim import corpora, models
import json


def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # 去除换行符和多余的空格
                page_text = page_text.replace('\n', ' ').replace('  ', ' ')
                text += page_text + ' '
    return text


def process_pdf_folder(folder_path):
    """ 处理文件夹中的所有PDF文件 """
    all_texts = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            text = extract_text_from_pdf(pdf_path)
            all_texts[file_name] = text
    return all_texts

def load_stopwords(stopwords_path):
    """ 加载停用词表 """
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file])
    return stopwords

def segment_text(text, stopwords):
    """ 对文本进行分词并去除停用词 """
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords]

def convert_to_serializable(obj):
    if isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


# 获取当前工作目录的路径
current_working_directory = os.getcwd()

# 构建到 'uploaded_files' 文件夹的路径
folder_path = os.path.join(current_working_directory, 'uploaded_files')
pdf_texts = process_pdf_folder(folder_path)

# 构建到 'stopwords_baidu.txt' 文件的路径
stopwords_path = os.path.join(folder_path, 'stopwords_baidu.txt')
stopwords = load_stopwords(stopwords_path)

# 对提取的PDF文本进行分词和去除停用词
segmented_texts = {}
for file_name, text in pdf_texts.items():
    segmented_texts[file_name] = segment_text(text, stopwords)

# 构建词袋模型
dictionary = corpora.Dictionary(segmented_texts.values())
corpus = [dictionary.doc2bow(text) for text in segmented_texts.values()]

# 使用LDA模型
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# 提取和打印主题
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# 修改后的主题信息整理
doc_topics_serializable = {}
for doc_id, topics in doc_topics.items():
    doc_topics_serializable[doc_id] = [(topic_id, float(prob)) for topic_id, prob in topics]

# 将主题信息保存为JSON文件
topics_json_path = os.path.join(folder_path, 'doc_topics.json')
with open(topics_json_path, 'w') as f:
    json.dump(doc_topics_serializable, f)
