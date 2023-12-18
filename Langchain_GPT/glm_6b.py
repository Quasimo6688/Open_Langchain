import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
MODEL_PATH = 'C:/Users/zk_sh/Desktop/GPT/Glm3_6b_int4'

# 检查 CUDA 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)


def generate_model_response(user_input):
    """
    生成模型的响应（根据实际情况调整此函数）。
    """
    # 将用户输入编码为模型可以理解的格式
    inputs = tokenizer.encode(user_input, return_tensors='pt').to(DEVICE)

    # 生成模型响应
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # 解码模型输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def chat_bot():
    print("欢迎使用聊天机器人！输入 '退出' 以结束对话。")

    while True:
        user_input = input("\n您：")
        if user_input.lower() == '退出':
            break

        response = generate_model_response(user_input)
        print("\n机器人：" + response)


if __name__ == "__main__":
    chat_bot()
