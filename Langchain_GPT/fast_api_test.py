import time
import logging
import numpy as np
import faiss
import os
import zhipuai
import queue
import json
import threading
from fastapi import FastAPI, HTTPException, Query, APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from starlette.responses import FileResponse
import random
from fastapi.responses import JSONResponse
import re
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

app = FastAPI()
session_dict = {}

class SessionState:
    def __init__(self):
        self.images_path = []
        self.is_ready = False
        self.output_queue = asyncio.Queue()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 设置zhipuai API密钥
zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"

# 全局变量定义
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_files_dir = os.path.join(script_dir, 'Embedding_Files')

# 初始化变量
embedding_path = ""
combined_text_path = ""
faiss_index_path = ""
pictures_index_path = ""

# 遍历目标文件夹，根据文件名进行分类
for filename in os.listdir(embedding_files_dir):
    if filename.endswith('.npy'):
        embedding_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.json'):
        if "collection" in filename:
            combined_text_path = os.path.join(embedding_files_dir, filename)
        elif "Pictures" in filename:
            pictures_index_path = os.path.join(embedding_files_dir, filename)
    elif filename.endswith('.index'):
        faiss_index_path = os.path.join(embedding_files_dir, filename)

# 加载向量知识库文件
def load_embeddings(embedding_path):
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
        logging.info("嵌入向量文件加载成功。")
        return embeddings
    else:
        logging.error(f"嵌入向量文件 {embedding_path} 未找到。")
        return None

# 初始化索引
def initialize_faiss(faiss_index_path):
    if os.path.exists(faiss_index_path):
        faiss_index = faiss.read_index(faiss_index_path)
        logging.info("FAISS索引加载成功。")
        return faiss_index
    else:
        logging.error("FAISS索引文件未找到。")
        return None

# 用户问题向量化
def get_query_vector(message):
    response = zhipuai.model_api.invoke(model="text_embedding", prompt=message)
    if response and response.get('success'):
        return response['data']['embedding']
    else:
        return None

# 通过索引向数据库比对返回内容
def search_in_faiss_index(query_vector, faiss_index, top_k=7):
    scores, indices = faiss_index.search(np.array([query_vector]), top_k)
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
            if page_number:
                page_numbers.append(int(page_number))
    session_state.images_path = extract_images_from_pages(page_numbers, pictures_index_path)
    return contents

def extract_images_from_pages(page_numbers, pictures_index_path):
    unique_pages = set(map(int, page_numbers))
    with open(pictures_index_path, 'r', encoding='utf-8') as file:
        pictures_map = json.load(file)
    matched_images = [item["image_path"] for key, item in pictures_map.items() if item["page"] in unique_pages]
    return matched_images

async def generate_response(prompt, session_state, headers):
    def process_streaming_output():
        logging.info("模型正常启动")
        try:
            response = zhipuai.model_api.sse_invoke(
                model="chatglm_turbo",
                prompt=prompt,
                temperature=0.2,
                incremental=True
            )
            logging.info(f"输入的最终提示词：{prompt}")
            logging.info("内部队列转录进行中")
            for event in response.events():
                if event.event == "add":
                    logging.info(f"Stream Output: {event.data}")
                    yield f"data: {json.dumps(event.data)}\n\n"
                elif event.event in ["error", "interrupted"]:
                    break
                else:
                    print(event.data)
        finally:
            session_state.is_ready = True

    return StreamingResponse(process_streaming_output(), headers=headers)


async def GLM_Streaming_response(message, session_state, headers):
    query_vector = get_query_vector(message)
    if query_vector is None:
        raise ValueError("无法获取查询向量，请检查问题并重试。")
    faiss_index = initialize_faiss(faiss_index_path)
    scores, indices = search_in_faiss_index(query_vector, faiss_index)
    logging.info(f"向量搜索正常启动")
    if indices is None:
        raise ValueError("搜索FAISS索引时出现问题，请重试。")
    combined_text = get_combined_text(indices, combined_text_path, pictures_index_path, session_state)
    logging.info(f"文本索引正常启动")
    if combined_text == "":
        raise ValueError("无法获取与查询相关的文本，请重试。")
    prompt = f"你是一名专业的飞行教练，使用中文和用户交流。你将提供精确且权威的答案给用户，深入问题所在，利用这些知识：{combined_text}。" \
             f"找到最合适的解答。如果答案在文档中，则会用文档的原文回答，并指出文档名及页码。若答案不在文档内，你将依据你的专业知识回答，" \
             f"并明确指出。你的回答将专注于航空领域的专业知识，旨在直接且有效地帮助用户解决问题。请确信，用户会获得与飞行训练和学习需求紧密" \
             f"相关的专业指导。回答的内容请排版为整齐有序的格式。" \
             f"请记住，安全永远是首要考虑，负责任的态度对于飞行至关重要。用户的问题是：{message}"
    return await generate_response(prompt, session_state, headers)


@app.get("/dialogue/")
async def dialogue(message: str, session_id: str):
    local_state = SessionState()
    session_dict[session_id] = local_state

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }

    response = await GLM_Streaming_response(message, local_state, headers)

    return response

@app.get("/return-image")
async def return_image(session_id: str):
    start_time = time.time()
    while True:
        if time.time() - start_time > 30:
            return JSONResponse({"images": []})
        if session_id in session_dict and session_dict[session_id].is_ready:
            matched_images = session_dict[session_id].images_path
            del session_dict[session_id]
            return JSONResponse({"images": matched_images})
        await asyncio.sleep(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

