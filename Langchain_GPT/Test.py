import numpy as np
from transformers import GPT2Tokenizer
import time
import os
import json
import logging  # 新增，用于日志功能
import configparser  # 新增，用于读取配置文件
import nltk
from nltk.corpus import wordnet
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.schema import (AIMessage,HumanMessage,SystemMessage)
from langchain.text_splitter import CharacterTextSplitter

# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
api_key_file_path = os.path.join(script_dir, 'api_key.txt')
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
metadata_path = os.path.join(script_dir, 'metadata.json')

# 初始化日志和配置
logging.basicConfig(level=logging.INFO)
config = configparser.ConfigParser()
config.read(os.path.join(script_dir, 'config.ini'))  # 使用相对路径读取配置文件

# 从配置文件中读取设置
api_key_file_path = config.get('Settings', 'api_key_file_path')
faiss_index_path = config.get('Settings', 'faiss_index_path')
embeddings_path = config.get('Settings', 'embeddings_path')
metadata_path = config.get('Settings', 'metadata_path')

# 初始化GPT-2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 初始化OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")
openai.api_key = api_key

# 初始化变量
REQUEST_DELAY_SECONDS = 2
DEBUG = False  # 用于控制是否打印日志

# 逐字符打印答案的函数
def print_char_by_char(answer):
    for char in answer:
        print(char, end='', flush=True)
        time.sleep(0.1)


# 创建文本嵌入的函数
def create_embedding(text):
    model_engine = "text-embedding-ada-002"
    response = openai.Embedding.create(
        model=model_engine,
        input=text,
    )
    return response['data'][0]['embedding']

# 问GPT-3的函数
def ask_gpt(prompt):
    model_engine = "gpt-3.5-turbo"
    response = openai.Completion.create(
        model=model_engine,
        prompt=prompt,
        max_tokens=3000
    )
    return response.choices[0].text.strip()

# 读取Faiss索引和嵌入
index = faiss.read_index(faiss_index_path)
embeddings = np.load(embeddings_path)

# 从JSON文件读取metadata
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 主逻辑
while True:
    print("=================会话开始(输入“退出”结束会话)==================\n")
    question = input("问：")
    total_tokens = 0  # 将这一行移动到这里

    # 控制日志打印
    if question.lower() == '打印日志':
        DEBUG = not DEBUG
        logging.info("日志打印已切换")
        continue


    if question.lower() == '退出':
        break

    question_embedding = create_embedding(question)
    D, I = index.search(np.array([question_embedding]), 5)

    matched_metadata = [metadata[i] for i in I[0] if i < len(metadata)]

    top_matches = []
    for item in matched_metadata:
        title = item.get('Title', '未知标题')  # 注意这里改为了'Title'
        context = item.get('Langchain_context', '未知内容')  # 注意这里改为了'Langchain_context'
        top_matches.append((title, context))

    if DEBUG:
        logging.info(f"Length of metadata: {len(metadata)}")
        logging.info(f"Max index in I[0]: {max(I[0])}")
        logging.info(f"Number of top matches: {len(top_matches)}")
        logging.info(f"Total tokens so far: {total_tokens}")
        for i, (title, context) in enumerate(top_matches):
            logging.info(f"Top match {i + 1}: Title: {title}, Context: {context}")

    cleaned_matches = []

    for title, context in top_matches:
        clean_context = context.replace('\n', ' ').strip()
        clean_content = f"{title} {clean_context}"
        tokens = tokenizer.encode(clean_content, add_special_tokens=False)
        if total_tokens + len(tokens) <= 1000:
            cleaned_matches.append(clean_content)
            total_tokens += len(tokens)
        else:
            break

    combined_text = " ".join(cleaned_matches)
    prompt = f"你是一个根据专业知识库回答问题的AI助手，优先参考以下双引号以内的开发文档内容进行回答，如不能回答或是开发文档内容和问题关联度过低则按照你的想法回答:\n“{combined_text}”\n\n注意语言的自然和专业，不要回答与问题无关的内容\n我的问题是：{question}"
    time.sleep(REQUEST_DELAY_SECONDS)
    answer = ask_gpt(prompt)

    print("答：")
    print_char_by_char(answer)
    print("\n=================会话结束(输入“退出”结束会话)==================\n")


