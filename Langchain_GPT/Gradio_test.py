import gradio as gr
import time


# 函数定义
def chat_function(message, chat_history):
    chat_history.append((message, ""))
    for i in range(20):
        time.sleep(0.2)
        chat_history[-1] = (message, f"自动消息 {i + 1}")
        yield "", chat_history

    return "", chat_history


# 界面定义
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    send = gr.Button("发送")

    send.click(chat_function, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_function, inputs=[msg, chatbot], outputs=[msg, chatbot])

# 启动界面
demo.queue().launch()
