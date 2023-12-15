import time
import logging
import numpy as np
import faiss
import os
import zhipuai
import queue
import json
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
import random
from fastapi.responses import JSONResponse
import re
from pydantic import BaseModel, Field


class SessionState:
    def __init__(self):
        self.images_path = []
        self.is_ready = False
        self.output_queue = queue.Queue()


class DialogueRequest(BaseModel):
    message: str
    session_id: str

session_dict = {}


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

def get_combined_text(indices, combined_text_path, pictures_index_path, session_state):
    with open(combined_text_path, 'r', encoding='utf-8') as file:
        combined_data = json.load(file)

    contents = []
    page_numbers = []
    for index in indices[0]:
        adjusted_index = str(index - 1)
        if adjusted_index in combined_data:
            text_entry = combined_data[adjusted_index]
            contents.append(text_entry.get("content", "No content found"))
            page_number = text_entry.get("page_number")
            if page_number is not None:
                page_numbers.append(int(page_number))
            else:
                logging.warning(f"找不到索引 {adjusted_index} 的页码")
        else:
            logging.warning(f"索引 {adjusted_index} 在 JSON 文件中未找到对应的文本内容")

    session_state.images_path = extract_images_from_pages(page_numbers, pictures_index_path)
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
def generate_response(prompt, sessionstate):
    def process_streaming_output():
        response = zhipuai.model_api.sse_invoke(
            model="chatglm_turbo",
            prompt=prompt,
            temperature=0.2,
            incremental=True
        )
        logging.info(f"输入的最终提示词：{prompt}")
        logging.info("内部队列转录进行中")
        try:
            for event in response.events():
                if event.event == "add":
                    logging.info(f"Stream Output: {event.data}")  # 添加此日志记录
                    session_state.output_queue.put(event.data)
                elif event.event in ["error", "interrupted"]:
                    break
                else:
                    print(event.data)
        finally:
            sessionstate.output_queue.put(None)  # 向队列发送结束信号
            sessionstate.is_ready = True

    # 启动处理线程
    threading.Thread(target=process_streaming_output).start()



def GLM_Streaming_response(message, session_state):
    # 用于获取用户问题的向量表示
    query_vector = get_query_vector(message)
    if query_vector is None:
        raise ValueError("无法获取查询向量，请检查问题并重试。")

    # 在FAISS索引中搜索并获取索引
    faiss_index = initialize_faiss(faiss_index_path)
    scores, indices = search_in_faiss_index(query_vector, faiss_index)
    if indices is None:
        raise ValueError("搜索FAISS索引时出现问题，请重试。")

    # 获取组合文本
    combined_text = get_combined_text(indices, combined_text_path, pictures_index_path, session_state)
    if combined_text == "":
        raise ValueError("无法获取与查询相关的文本，请重试。")

    # 构建提示信息并生成响应
    prompt = f"你是一名专业的飞行教练，使用中文和用户交流。你将提供精确且权威的答案给用户，深入问题所在，利用这些知识：{combined_text}。" \
             f"找到最合适的解答。如果答案在文档中，则会用文档的原文回答，并指出文档名及页码。若答案不在文档内，你将依据你的专业知识回答，并明确指出。" \
             f"你的回答将专注于航空领域的专业知识，旨在直接且有效地帮助用户解决问题。请确信，用户会获得与飞行训练和学习需求紧密相关的专业指导。回答" \
             f"的内容请排版为整齐有序的格式。请记住，安全永远是首要考虑，负责任的态度对于飞行至关重要。用户的问题是：{message}"

    generate_response(prompt, session_state)

    return session_state.output_queue


##################Fast_API接口封装#######################

@app.get("/dialogue/")
async def dialogue(dialogue_request: DialogueRequest):
    try:
        # 创建新的会话状态并存储在 session_dict 中
        local_state = SessionState()
        session_dict[dialogue_request.session_id] = local_state

        # 生成响应并获取输出队列
        output_queue = GLM_Streaming_response(dialogue_request.message, local_state)

        async def event_generator():
            while True:
                data = output_queue.get()
                if data is None:  # 检查队列结束信号
                    break
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
async def return_image(session_id: str):
    if session_id not in session_dict or not session_dict[session_id].is_ready:
        raise HTTPException(status_code=400, detail="Session not ready or not found.")

    matched_images = session_dict[session_id].images_path

    # 删除会话状态以避免内存泄漏
    del session_dict[session_id]

    return JSONResponse({"images": matched_images})




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




