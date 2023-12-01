import logging
import zhipuai
import gradio as gr

# 初始化 logging 和 zhipuai API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"

# 主要函数：React_Agent_Chain
def React_Agent_Chain(message):
    # 构建固定提示词模板
    step_1 = f"请判断问题是否含有一个问题，只需输出’是‘或’否‘一个字。问题：{message}"

    logging.info(f"执行代理分析流程……")
    response = zhipuai.model_api.invoke(
        model="chatglm_turbo",
        prompt=step_1,
        temperature=0.0,
        incremental=True
    )
    logging.info(f"用户提问：{message}，分析用户问题类型……")

    # 从响应中提取模型的回答
    if response and 'data' in response and 'choices' in response['data'] and response['data']['choices']:
        model_answer = response['data']['choices'][0]['content']  # 提取第一个回答的内容
        return model_answer
    else:
        # 如果响应中没有找到预期的数据，返回默认回答或错误信息
        return "抱歉，这个问题或许超出了知识库内容范围。"

Test_UI = gr.ChatInterface(React_Agent_Chain)

if __name__ == "__main__":
    Test_UI.launch(share=True, inbrowser=True)
