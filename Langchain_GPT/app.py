#导入初始加载模块：langchain_faiss_setup
from langchain_faiss_setup import initialize_langchain_and_faiss，load_api_key
#导入GPT聊天模块
from gpt3_integration_langchain import initialize_langchain_gpt3, ask_gpt3_langchain

#初始化Langchain和Faiss
openai_embeddings, faiss_index, embeddings = initialize_langchain_and_faiss()
加载GPT-API密钥
api_key = load_api_key()
# 初始化 GPT-3.5
model_engine = initialize_langchain_gpt3(api_key)
# 使用 GPT-3.5 进行问答
response = ask_gpt3("What is the meaning of life?", model_engine)