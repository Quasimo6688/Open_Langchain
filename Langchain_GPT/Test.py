import numpy as np
import time
import os
import json
import logging  # 用于日志功能
import configparser  # 用于读取配置文件
import nltk
import getpass

from nltk.corpus import wordnet
from transformers import GPT2Tokenizer
# Langchain 相关导入
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain import LLMChain, PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# 假设这些是你可能需要的 Langchain Agents 和其他组件
from langchain.agents import OpenAIFunctionsAgent# 用于与语言模型交互
from langchain.retrieval import Retrieval  # 用于数据检索
from langchain.chains import Chains  # 用于构建调用序列
from langchain.memory import Memory  # 用于在链的运行之间保持应用状态
from langchain.callbacks import Callbacks  # 用于记录和流式传输任何链的中间步骤

#谷歌搜索功能加载项
from langchain. agents import load_tools
from langchain. agents import initialize_agent
from langchain. llms import OpenAI
from langchain. agents import AgentType

import langchain_gradio_chat_interface #导入Gradio模块

langchain_gradio_chat_interface.start_UI() #Gradio模块界面启动函数

# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
api_key_file_path = os.path.join(script_dir, 'api_key.txt')
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
metadata_path = os.path.join(script_dir, 'metadata.json')


# 初始化日志和配置
logging.basicConfig(level=logging.INFO)

# 初始化GPT-2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#初始化Open_AI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

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

# 在你的函数和逻辑中使用这些组件
def ask_gpt_with_agent(prompt):
    # 使用不同的 Agents 和组件来获取答案，根据需要
    response = complex_agent.ask(prompt, knowledge_base)
    return response

# 读取Faiss索引和嵌入
index = faiss.read_index(faiss_index_path)
embeddings = np.load(embeddings_path)

# 从JSON文件读取metadata
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 主逻辑
while True:
    question = input("问：")

    total_tokens = 0  #

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
    # 使用新的 ask_gpt_with_agent 函数
    answer = ask_gpt_with_agent(prompt)

    print("答：", answer)
    print_char_by_char(answer)
    print("\n=================会话结束(输入“退出”结束会话)==================\n")


