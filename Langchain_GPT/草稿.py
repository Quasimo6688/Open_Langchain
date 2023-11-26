import time
import logging
import numpy as np
import faiss
import os
import zhipuai
import gradio as gr
import queue
import json
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 设置zhipuai API密钥
zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"


#全局变量定义：
    # 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_files_dir = os.path.join(script_dir, 'Embedding_Files')

# 初始化变量
embedding_path = ""
combined_text_path = ""
faiss_index_path = ""

# 遍历目标文件夹，根据文件后缀进行分类
for filename in os.listdir(embedding_files_dir):
    if filename.endswith('.npy'):
        embedding_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.json'):
        combined_text_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.index'):
        faiss_index_path = os.path.join(embedding_files_dir, filename)


# 创建响应队列
response_queue = queue.Queue()


  #加载向量知识库文件
def load_embeddings(embedding_path):
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
        logging.info("嵌入向量文件加载成功。")
        return embeddings
    else:
        logging.error(f"嵌入向量文件 {embedding_path} 未找到。")
        return None
  #初始化索引
def initialize_faiss(faiss_index_path):
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
        logging.info("FAISS索引加载成功。")
        return faiss_index
    else:
        logging.error("FAISS索引文件未找到。")
        return None
    return faiss_index

  #正式的问答流程：**********************************************************************

  #用户问题向量化
def get_query_vector(message):
    response = zhipuai.model_api.invoke(model="text_embedding", prompt=message)

    # 检查响应是否成功
    if response and response.get('success'):
        # 打印 token 消耗信息
        usage = response['data'].get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', '未知')
        completion_tokens = usage.get('completion_tokens', '未知')
        total_tokens = usage.get('total_tokens', '未知')
        logging.info(f"Token 消耗信息 - 提示 Tokens: {prompt_tokens}, 完成 Tokens: {completion_tokens},"
                     f" 总 Tokens: {total_tokens}")

        # 返回 embedding
        return response['data']['embedding']
    else:
        # 如果响应不成功，仅返回 None
        return None

  #通过索引向数据库比对返回内容
def search_in_faiss_index(query_vector, faiss_index, top_k=7):
    # 在FAISS索引中搜索
    scores, indices = faiss_index.search(np.array([query_vector]), top_k)
    logging.info(f"FAISS搜索得分: {scores}, 索引: {indices}")
    return scores, indices

def get_combined_text(indices, combined_text_path):
    # 从合并的 JSON 文件中读取内容
    with open(combined_text_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 根据索引获取相应的文本块
    text_blocks = [data[str(index)] for index in indices[0]]  # 正常状态下每个索引对应一个文本块
    logging.info(f"通过索引找到的文本块: {text_blocks}")

    # 拼接文本块
    combined_result = "\n".join(text_blocks)
    return combined_result

  #将返回问题加工成最终的模型提问发送请求等待返回
def generate_response(prompt):
    output_queue = queue.Queue()
    def process_streaming_output():
        # 使用zhipuai聊天模型生成回答，开启新线程并进行流式输出
        response = zhipuai.model_api.sse_invoke(
            model="chatglm_turbo",
            prompt=prompt,
            temperature=0.2,
            incremental=True
        )  # 增量返回，否则为全量返回
        logging.info(prompt)
        try:
            for event in response.events():
                if event.event == "add":
                    #这里向函数内的队列写入输出内容
                    output_queue.put(event.data)
                elif event.event in ["error", "interrupted", "finish"]:
                    break
        finally:
            output_queue.put(None)  # 向队列发送结束信号

    # 启动处理线程
    threading.Thread(target=process_streaming_output).start()

    return output_queue


def Gr_UI(message):

    # 获取查询的向量转化结果
    query_vector = get_query_vector(message)
    if query_vector is None:
        # 如果无法获取查询向量，则返回错误信息
        return "无法获取查询向量，请检查问题并重试。"

    # 在FAISS索引中搜索并获取索引
    scores, indices = search_in_faiss_index(query_vector, faiss_index)
    if indices is None:
        # 如果无法在FAISS索引中搜索，则返回错误信息
        return "搜索FAISS索引时出现问题，请重试。"

    # 获取组合文本
    combined_text = get_combined_text(indices, combined_text_path)
    if combined_text == "":
        # 如果组合文本为空，则返回错误信息
        return "无法获取与查询相关的文本，请重试。"

    # 构建提示信息
    prompt = f"你是一名专业的飞行教练，使用中文和用户交流。你将提供精确且权威的答案给用户，深入问题所在，利用这些知识：{combined_text}。" \
             f"找到最合适的解答。如果答案在文档中，则会用文档的原文回答，并指出文档名及页码。若答案不在文档内，你将依据你的专业知识回答，并明确指出。" \
             f"你的回答将专注于航空领域的专业知识，旨在直接且有效地帮助用户解决问题。请确信，用户会获得与飞行训练和学习需求紧密相关的专业指导。" \
             f"请记住，安全永远是首要考虑，负责任的态度对于飞行至关重要。用户的问题是：{message}"
    output_queue = generate_response(prompt)
    finish_answer = ""
    # 从队列中获取回应并返回
    while True:
        response = output_queue.get()
        if response is None:
            logging.info("接收到终止信号，输出完成。")
            break  # 接收到结束信号，退出循环
        finish_answer = (f"{finish_answer + response}")
        time.sleep(0.1)
        yield finish_answer

    # 清空队列（确保这个函数已经定义）

    logging.info("流程结束，清空队列")


#函数调用：
embeddings = load_embeddings(embedding_path)
    # 初始化FAISS索引
faiss_index = initialize_faiss(faiss_index_path)

    # 启动用户界面
#Start_UI = gr.ChatInterface(Gr_UI).queue()
#Start_UI.launch(share=True, inbrowser=True)
