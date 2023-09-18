from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
#初始化函数
def initialize_langchain_gpt3(api_key):
    chat_model = ChatOpenAI(api_key=api_key, model_engine="gpt-3.5-turbo")
    return chat_model

def ask_gpt3_langchain(prompt, chat_model):
    response = chat_model.ask(prompt)
    return response.text.strip()
