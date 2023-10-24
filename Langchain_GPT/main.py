import os
import logging
import time
import openai
# Langchain 相关导入
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
# 导入新的模块
from model_manager import initialize_model
from gr_interface import start_UI
from state_manager import get_state, update_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 相对位置加载配置文件
script_dir = os.path.dirname(os.path.abspath(__file__))
api_key_file_path = os.path.join(script_dir, 'key.txt')

# 自动填写OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")
#传递key到模型模块
model_info = initialize_model(api_key)

# 启动Gradio界面
start_UI(model_info, get_state())


