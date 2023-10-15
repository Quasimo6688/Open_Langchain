import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from model_manager import get_response_from_model
from state_manager import get_state, update_state
from langchain.schema import HumanMessage, SystemMessage, AIMessage
#界面视觉设定：
theme = gr.themes.Glass().set(
    body_background_fill='*primary_300',
    block_background_fill='*primary_100',
    block_border_width='3px',
    chatbot_code_background_color='*primary_100',
    button_large_padding='*spacing_xs',
    button_large_radius='*radius_md',
    button_large_text_weight='500',
    button_small_radius='*radius_sm',
    button_small_text_size='*text_xs'
)

# 定义布局和组件
with gr.Blocks(theme=theme) as ui:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="聊天机器人", bubble_full_width=False, container=True, height=400) #avatar_images 元组[str |路径 |无，str |路径 |无] |没有默认值：无;用户和机器人的两个头像图像路径或 URL 的元组（按此顺序）。传递“无”，以
            msg = gr.Textbox(label="输入消息", placeholder="您好，我是一个专业数据库问答助手，请在这里输入问题……", lines=3)
            with gr.Row():
                with gr.Column():
                    temperature_UI = gr.Slider(label="温度", minimum=0, maximum=1, step=0.1, scale=4)
                with gr.Column():
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot], value="清除", min_width=88)
                        refresh = gr.Button("刷新", min_width=88)
                        send = gr.Button("发送", min_width=88)
            gr.Markdown("代理模块提示词控制台.")
            with gr.Tab("用户提问提示词"):
                template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    usr_clear = gr.Button("清空模板", min_width=88)
                    usr_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("分析器提示词"):
                analyzer_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    analyzer_clear = gr.Button("清空模板", min_width=88)
                    analyzer_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("工具代理提示词"):
                tool_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    tool_clear = gr.Button("清空模板", min_width=88)
                    tool_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("搜索代理提示词"):
                search_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    search_clear = gr.Button("清空模板", min_width=88)
                    search_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("回答模板提示词"):
                answer_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    answer_clear = gr.Button("清空模板", min_width=88)
                    answer_Upload = gr.Button("保存模板", min_width=88)
            with gr.Tab("临时提示词"):
                temporary_template = gr.Textbox(label="提示词模板", placeholder="在这里输入提示词，提交后生效", lines=2)
                with gr.Row():
                    temporary_clear = gr.Button("清空模板", min_width=88)
                    temporary_Upload = gr.Button("保存模板", min_width=88)

        with gr.Column():
            example_image = gr.Image(label="示例图像")
            gr.Markdown("日志调试台.")
            with gr.Tab("Langchain日志"):
                log_output_box = gr.Textbox(label="Langchain日志", lines=16)
            with gr.Tab("代理反应"):
                agent_output_box = gr.Textbox(label="代理反应", lines=16)


    def chat_function(message, chat_history, temperature, template):
        global_state = get_state()
        global_state.text_input = message  # 更新状态的值
        global_state.bot_message = None
        system_msg = [HumanMessage(content=global_state.text_input)]
        gpt_response = get_response_from_model(global_state.llm_function, system_msg)  # 使用传递的 chat 实例
        global_state.finish_answer = gpt_response.content
        global_state.module_template = template

        global_state.agent_output_str += f"这是代理的输出: {global_state.finish_answer}\n"
        global_state.log_output_str += f"用户提问: {message}, 系统最终回答: {gpt_response.content}\n"

        # 更新chat_history
        chat_history.append((message, global_state.finish_answer))

        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, '示例图像', fontsize=12, ha='center')
        plt.axis('off')

        # 为提问创建一个图形
        plt.figure(figsize=(4, 4))
        question_img = Image.open("Question_image.png")
        plt.imshow(question_img)
        plt.axis('off')
        plt.savefig("question_image.png")

        return "", chat_history, global_state.agent_output_str, global_state.log_output_str, (
            "question_image.png")

    # 绑定事件处理函数到按钮
    send.click(chat_function, inputs=[msg, chatbot, temperature_UI, template],outputs=[msg, chatbot, agent_output_box, log_output_box, example_image])


    # 绑定函数到文本框和聊天机器人组件
    msg.submit(chat_function, [msg, chatbot, temperature_UI, template], [msg, chatbot, agent_output_box, log_output_box, example_image])

def start_UI(func, state_instance):
    global global_state
    global_state = state_instance
    global_state.llm_function = func
    ui.launch(share=True, inbrowser=True)
