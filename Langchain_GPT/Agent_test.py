# coding=utf-8
import gradio as gr
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import langchain_gradio_chat_interface as ci
# 输入和输出变量
input_query = None  # 用于存储从主程序或代理接收的查询
output_summaries = None  # 用于存储搜索结果的摘要，以便主程序或代理调用
#提问方法
def greet(query):
    global output_summaries
    wrapper = WikipediaAPIWrapper()
    output_summaries = wrapper.run(query)
    ci.chat_function()
    return "回答:" + output_summaries + "！"

# 创建一个Gradio界面，将greet函数作为输入和输出函数传递给它
iface = gr.Interface(fn=greet, inputs="text", outputs="text")

# 启动Gradio应用程序
iface.launch()