# pip install zhipuai 请先在终端进行安装

import zhipuai

zhipuai.api_key = "1a21c86a3aa8f435250194b3dc9dc6b8.2Aov2pnPfNB7lLPi"
response = zhipuai.model_api.sse_invoke(
    model="chatglm_turbo",
    prompt= [{"role":"user","content":"你是一个智能代理，通过识别其他代理发送的指令进行下一步操作，当收到消息开头为：非问候语，这四个字时，"
                                      "你要根据消息后面的内容判断该使用何种工具辅助回答问题，比如当收到的消息为：非问候语，关于空气动力学你"
                                      "都知道些什么？，那么你应该知道实际的问题是：关于空气动力学你都知道些什么?针对这个问题，你要思考使用"
                                      "哪一个工具辅助回答是最有必要的。你能调用的工具有：1.魔法精专手册2.飞行员知识手册3Langchain官方文档"
                                      "4.Gradio官方网文档5.视频搜索引擎6.维基百科。确定该使用哪一个工具后，直接回答这个工具名称，不要回答其他任何"
                                      "内容！收到的消息：非问候语，如何练习中级大火球术？"}],
    temperature= 0.9,
    top_p= 0.7,
    incremental=True
)

for event in response.events():
    if event.event == "add":
        print(event.data, end="")
    elif event.event == "error" or event.event == "interrupted":
        print(event.data, end="")
    elif event.event == "finish":
        print(event.data)
        print(event.meta, end="")
    else:
        print(event.data, end="")