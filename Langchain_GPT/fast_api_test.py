import time
import logging
import numpy as np
import faiss
import os
import zhipuai
import queue
import json
import threading
import state_manager
from state_manager import shared_output, Images_path, glm_chat_history, If_run
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
from pydantic import BaseModel
import random
from fastapi.responses import JSONResponse
import re

class DialogueRequest(BaseModel):
    message: str

app = FastAPI()
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
pictures_index_path = ""  # 新增变量，用于图片索引文件

# 遍历目标文件夹，根据文件名进行分类
for filename in os.listdir(embedding_files_dir):
    if filename.endswith('.npy'):
        embedding_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.json'):
        # 根据文件名中的关键词来区分不同的 JSON 文件
        if "collection" in filename:  # 假设向量集合文本文件包含 "collection" 关键词
            combined_text_path = os.path.join(embedding_files_dir, filename)
        elif "Pictures" in filename:  # 假设图片索引文件包含 "pictures" 关键词
            pictures_index_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.index'):
        faiss_index_path = os.path.join(embedding_files_dir, filename)



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

def get_combined_text(indices, combined_text_path, pictures_index_path):
    # 从合并的 JSON 文件中读取内容
    with open(combined_text_path, 'r', encoding='utf-8') as file:
        combined_data = json.load(file)

    contents = []
    page_numbers = []
    for index in indices[0]:  # 假设 indices 是一维数组
        adjusted_index = str(index - 1)  # 调整索引（Python索引从0开始）

        if adjusted_index in combined_data:
            text_entry = combined_data[adjusted_index]
            contents.append(text_entry.get("content", "No content found"))
            page_number = text_entry.get("page_number")
            if page_number is not None:
                page_numbers.append(int(page_number))  # 确保是整数
            else:
                logging.warning(f"找不到索引 {adjusted_index} 的页码")
        else:
            logging.warning(f"索引 {adjusted_index} 在 JSON 文件中未找到对应的文本内容")

    state_manager.Images_path = extract_images_from_pages(page_numbers, pictures_index_path)
    state_manager.If_run = True
    return contents


def extract_images_from_pages(page_numbers, pictures_index_path):
    print("关联的页码", page_numbers)
    # 去除重复的页码
    unique_pages = set(page_numbers)
    print("去重后的页码", unique_pages)

    # 确保 unique_pages 中的所有元素都是整数
    unique_pages = set(map(int, unique_pages))

    # 读取 Pictures_map.json
    with open(pictures_index_path, 'r', encoding='utf-8') as file:
        pictures_map = json.load(file)

    # 寻找匹配的条目
    matched_images = []
    for key, item in pictures_map.items():
        # 确保比较时类型一致
        if item["page"] in unique_pages:
            matched_images.append(item["image_path"])

    print("匹配到的路径", matched_images)
    return matched_images



  #将返回问题加工成最终的模型提问发送请求等待返回
def generate_response(prompt):

    #接收问题并输出一个队列
    def process_streaming_output():
        # 使用zhipuai聊天模型生成回答，开启新线程并进行流式输出
        response = zhipuai.model_api.sse_invoke(
            model="chatglm_turbo",
            prompt=prompt,
            temperature=0.2,
            incremental=True
        )  # 增量返回，否则为全量返回
        logging.info(f"输入的最终提示词：{prompt}")
        logging.info(f"全局队列转录进行中")
        try:
            for event in response.events():
                if event.event == "add":
                    #这里向函数内的队列写入输出内容
                    shared_output.put(event.data)
                elif event.event in ["error", "interrupted"]:
                    break
                else:
                    print(event.data)
        finally:
            shared_output.put(None)  # 向队列发送结束信号#

    # 启动处理线程
    threading.Thread(target=process_streaming_output).start()

    return shared_output


def GLM_Streaming_response(message):
    logging.info(f"模型启动程序调用正常")

    # 初始化向量和FAISS索引
    load_embeddings(embedding_path)
    faiss_index = initialize_faiss(faiss_index_path)

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
    combined_text = get_combined_text(indices, combined_text_path, pictures_index_path)
    if combined_text == "":
        # 如果组合文本为空，则返回错误信息
        return "无法获取与查询相关的文本，请重试。"

    # 构建提示信息
    prompt = f"你是一名专业的飞行教练，使用中文和用户交流。你将提供精确且权威的答案给用户，深入问题所在，利用这些知识：{combined_text}。" \
             f"找到最合适的解答。如果答案在文档中，则会用文档的原文回答，并指出文档名及页码。若答案不在文档内，你将依据你的专业知识回答，并明确指出。" \
             f"你的回答将专注于航空领域的专业知识，旨在直接且有效地帮助用户解决问题。请确信，用户会获得与飞行训练和学习需求紧密相关的专业指导。回答" \
             f"的内容请排版为整齐有序的格式" \
             f"请记住，安全永远是首要考虑，负责任的态度对于飞行至关重要。用户的问题是：{message}"

    output = generate_response(prompt)
    return output


  ##################Fast_API接口封装#######################

@app.get("/dialogue/")
async def dialogue(dialogue_request: DialogueRequest):
    try:
        output_queue = GLM_Streaming_response(dialogue_request.message)

        accumulated_output = ""  # 用于累积输出内容的变量

        async def event_generator():
            nonlocal accumulated_output
            while True:
                data = output_queue.get()
                print(data)
                if data is None:  # 检查队列结束信号
                    break

                accumulated_output += data  # 累积接收到的数据
                print(f"\r流式输出数据：{accumulated_output}", end="")  # 在同一行更新显示所有累积的数据
                yield f"data: {json.dumps(data)}\n\n"

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }

        return StreamingResponse(event_generator(), headers=headers)

    except Exception as e:
        logging.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/return-image")
async def return_image(name: str = None):
    while not state_manager.If_run:
        # 如果 If_run 为 False，则等待
        print("程序暂停，等待中...")
        time.sleep(1)

        # 当 If_run 为 True，退出循环并继续执行
    matched_images = state_manager.Images_path

    # 在返回之前将 If_run 设置为 False
    state_manager.If_run = False

    # 返回图片列表
    return JSONResponse({"images": matched_images})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




