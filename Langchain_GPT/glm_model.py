# glm_model.py

import queue
import threading
import logging
import zhipuai
from state_manager import shared_output
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 配置日志记录器
logger = logging.getLogger(__name__)

class CustomStreamingCallbackGLM(StreamingStdOutCallbackHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.full_response = ""

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)
        self.full_response += token
        logger.info(f"模型输出: {self.full_response}")

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)
        logger.info(f"模型输出完成: {self.full_response}")
        self.full_response = ""


def initialize_model_GLM(api_key, model_code="chatglm_turbo"):
    zhipuai.api_key = api_key
    model_instance = zhipuai.model_api.sse_invoke
    model_params = {
        "model": model_code,
        "temperature": temperature,
        # 其他参数根据需要添加
    }
    return model_instance, model_params


def process_streaming_output_GLM(streaming_buffer):
    while True:
        token = streaming_buffer.get()
        if token is None:
            logger.info("转录器接收到结束信号")
            shared_output.put(token)
            break
        time.sleep(0.05)
        shared_output.put(token)
        logger.info(f"转录进行中: {token}")


def get_response_from_model_GLM(chat_instance, system_msg):
    streaming_buffer = queue.Queue()
    thread = threading.Thread(target=process_streaming_output_GLM, args=(streaming_buffer,))
    thread.start()

    callbacks = [CustomStreamingCallbackGLM(streaming_buffer)]
    chat_instance.callbacks = callbacks
    chat_instance(messages=system_msg)

    # 没有while循环，因为处理输出的逻辑已经在process_streaming_output_GLM函数中处理
