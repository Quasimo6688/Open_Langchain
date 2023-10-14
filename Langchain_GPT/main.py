import numpy as np
import time
import os
import logging  # 用于日志功能
import configparser  # 用于读取配置文件
import nltk
import getpass
import openai
import json

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

#谷歌搜索功能加载项
from langchain. agents import load_tools
from langchain. agents import initialize_agent
from langchain. llms import OpenAI
from langchain. agents import AgentType
from langchain.memory import ConversationBufferMemory #内存记忆模块
import langchain_gradio_chat_interface #导入Gradio模块

global_finish_answer = "" #声明全局返回变量，这里用来存储模型最终回答,问题输入的全局变量在UI模块中已经声明。
# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
api_key_file_path = os.path.join(script_dir, 'key.txt') #存储OPAI_API_KEY的文档
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')#faiss索引文件
embeddings_path = os.path.join(script_dir, 'embeddings.npy')#Langchain知识库嵌入文件
metadata_path = os.path.join(script_dir, 'metadata.json')#知识库元数据

# 初始化日志和配置
logging.basicConfig(level=logging.INFO)

# 初始化GPT-2分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# 自动填写OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")
openai.api_key = api_key
#初始化Open_AI
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
# 初始化变量
REQUEST_DELAY_SECONDS = 2
DEBUG = False  # 用于控制是否打印日志

def llm_to_UI():
    global global_finish_answer
    global global_text_input
    # 创建一个系统消息
    system_msg = SystemMessage(content="你是一个聊天助手，使用中文进行交流.")
    gpt_response = chat([HumanMessage(content=global_text_input)])
    # 将模型的答案存储在全局变量中
    global_finish_answer = gpt_response.content
 # 返回LLM生成的文本内容
    return global_finish_answer

#执行Gradio模块的界面启动函数
langchain_gradio_chat_interface.start_UI(llm_to_UI)
