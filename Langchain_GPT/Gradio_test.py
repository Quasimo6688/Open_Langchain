import gradio as gr
import time
import random
import logging
import queue
import threading
from model_manager import get_response_from_model
from state_manager import get_state, update_state
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from 草稿 import generate_random_numbers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 全局开关变量
streaming_active = True

# 模拟流式输入函数
def simulate_streaming_input():
    global streaming_active
    response_text = ""
    for number in generate_random_numbers():  # 调用新模块的函数
        if not streaming_active:  # 检查开关状态
            break
        response_text += number  # 直接将新的随机数和分号添加到response_text
        yield response_text
    streaming_active = False  # 在模拟结束后关闭开关

# 函数定义
def chat_function(message, chat_history):
    global streaming_active
    streaming_active = True  # 重置状态开关
    chat_history.append((message, ""))
    for simulated_response in simulate_streaming_input():
        if not streaming_active:  # 检查开关状态
            break
        chat_history[-1] = (message, simulated_response)
        yield "", chat_history
    return "", chat_history

# 界面定义
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    send = gr.Button("发送")

    send.click(chat_function, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_function, inputs=[msg, chatbot], outputs=[msg, chatbot])

# 启动界面
demo.queue().launch()
