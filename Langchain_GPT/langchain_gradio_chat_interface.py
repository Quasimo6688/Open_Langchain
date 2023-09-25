import gradio as gr
import matplotlib.pyplot as plt


def chat_function(message, history):
    agent_output = "这是代理的回应"
    log_output = "这是代理的工作流程和日志"

    plt.figure(figsize=(4, 4))
    plt.text(0.5, 0.5, '示例图像', fontsize=12, ha='center')
    plt.axis('off')
    plt.savefig("example.png")

    return agent_output, log_output, "example.png"


iface = gr.Interface(
    fn=chat_function,
    inputs=[
        gr.Textbox(label="输入消息"),
        gr.Textbox(label="历史记录")
    ],
    outputs=[
        gr.Textbox(label="代理回应"),
        gr.Textbox(label="代理日志"),
        gr.Image(label="示例图像")
    ],
    live=True
)

iface.launch()
