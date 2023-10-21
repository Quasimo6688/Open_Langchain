import openai
import time
import queue
import logging
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

# 配置日志记录器
logger = logging.getLogger(__name__)

class CustomStreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.full_response = ""  # 用于存储整个响应的字符串

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)  # 将新令牌放入队列
        self.full_response += token  # 将新令牌附加到完整响应字符串
        logger.info(f"Streaming Response: {self.full_response}")  # 实时记录流式输出的当前状态

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)  # 当输出结束时，将 None 放入队列
        logger.info(f"Complete Response: {self.full_response}")  # 记录整个响应
        self.full_response = ""  # 重置完整响应字符串






def initialize_model(api_key, model_name="gpt-3.5-turbo", temperature=0.5, streaming=True):
    streaming_buffer = queue.Queue()
    callbacks = [CustomStreamingCallback(streaming_buffer)]
    return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key, streaming=streaming,
                      callbacks=callbacks)

def get_response_from_model(chat_instance, system_msg):
    # 创建一个队列来存放模型的流式输出
    streaming_buffer = queue.Queue()
    # 使用自定义的回调处理器
    callbacks = [CustomStreamingCallback(streaming_buffer)]
    # 传递回调处理器到模型中
    chat_instance.callbacks = callbacks
    # 开始请求模型并获取响应
    chat_instance(messages=system_msg)
    # 使用循环从队列中获取模型的流式输出
    while True:
        token = streaming_buffer.get()
        if token is None:  # 检查结束信号
            break
        time.sleep(0.05)
        yield token
