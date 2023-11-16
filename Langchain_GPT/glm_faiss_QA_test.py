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
    # 定义嵌入向量文件和文本文件的路径
embeddings_paths = [
    os.path.join(script_dir, 'Embedding_Files', '航空知识手册全集A.pdf_vectors.npy'),
    os.path.join(script_dir, 'Embedding_Files', '航空知识手册全集B.pdf_vectors.npy'),
    os.path.join(script_dir, 'Embedding_Files', '运动驾驶员执照理论 考试知识点(试行) 缩水.pdf_vectors.npy')
    ]
text_files = [
    os.path.join(script_dir, 'Embedding_Files', '航空知识手册全集A.pdf_collection.json'),
    os.path.join(script_dir, 'Embedding_Files', '航空知识手册全集B.pdf_collection.json'),
    os.path.join(script_dir, 'Embedding_Files', '运动驾驶员执照理论 考试知识点(试行) 缩水.pdf_collection.json')
    ]
# 定义合并后的文本文件路径
combined_text_path = os.path.join(script_dir, 'Embedding_Files', 'combined_text_file.txt')
# faiss索引文件路径
faiss_index_path = os.path.join(script_dir, 'Embedding_Files', 'faiss_glm.index')

# 创建响应队列
response_queue = queue.Queue()


  #加载向量知识库文件
def load_embeddings(embeddings_paths):
    all_embeddings = []
    for path in embeddings_paths:
        try:
            if os.path.exists(path):
                embeddings = np.load(path)
                all_embeddings.append(embeddings)
            else:
                logging.error(f"嵌入向量文件 {path} 未找到。")
                return None
        except Exception as e:
            logging.error(f"加载嵌入向量文件 {path} 时出错: {e}")
            return None
    return np.concatenate(all_embeddings, axis=0)


def combine_json_files(file_paths, combined_file_path):
    combined_data = {}  # 创建一个字典来存储合并后的数据
    next_index = 0  # 下一个可用的行号

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 加载 JSON 文件的内容
            for key, value in data.items():
                combined_data[str(next_index)] = value  # 使用递增的行号作为键
                next_index += 1  # 增加下一个行号

    # 将合并后的数据写入合并文件
    with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
        json.dump(combined_data, combined_file, ensure_ascii=False, indent=4)
   #加载索引文件
def initialize_faiss(faiss_index_path):
    if os.path.exists(faiss_index_path):
        # 使用 faiss 库直接读取索引
        faiss_index = faiss.read_index(faiss_index_path)
        logging.info("FAISS索引加载成功。")
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

  #数据库比对返回内容
def search_in_faiss_index(query_vector, faiss_index, top_k=3):
    # 在FAISS索引中搜索
    scores, indices = faiss_index.search(np.array([query_vector]), top_k)
    return scores, indices

def get_combined_text(indices, combined_text_path):
    # 从合并的 JSON 文件中读取内容
    with open(combined_text_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 根据索引获取相应的文本块
    text_blocks = [data[str(index)] for index in indices[0]]  # 假设每个索引对应一个文本块

    # 拼接文本块
    combined_result = "\n".join(text_blocks)
    return combined_result

  #将返回问题加工成最终的模型提问发送请求等待返回
def generate_response(prompt):
    output_queue = queue.Queue()
    def process_streaming_output():
        # 使用zhipuai聊天模型生成回答，这里省略了具体的调用细节
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
                    output_queue.put(event.data)
                elif event.event in ["error", "interrupted", "finish"]:
                    break
        finally:
            output_queue.put(None)  # 发送结束信号

    # 启动处理线程
    threading.Thread(target=process_streaming_output).start()

    return output_queue


def Gr_UI(message, history):

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
embeddings = load_embeddings(embeddings_paths)
    # 初始化FAISS索引
faiss_index = initialize_faiss(faiss_index_path)
    # 执行合并文本文件
combine_json_files(text_files, combined_text_path)
    # 启动用户界面
Start_UI = gr.ChatInterface(Gr_UI).queue()
Start_UI.launch(share=True, inbrowser=True)
