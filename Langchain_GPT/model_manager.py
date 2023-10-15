import openai
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

def initialize_model(api_key, model_name="gpt-3.5-turbo", temperature=0.5, streaming=True):
    return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key, streaming=streaming, callbacks=[StreamingStdOutCallbackHandler()])

def get_response_from_model(chat_instance, system_msg):
    return chat_instance(messages=system_msg)
