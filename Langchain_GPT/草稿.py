import requests

# 获取 API 地址
url = "https://api.langchain.asia/docs/v0.0.291/search"

# 构建请求参数
params = {"q": "你好，你都能做些什么？"}

# 发送请求
response = requests.get(url, params=params)

# 处理响应
results = response.json()

# 打印结果
for result in results:
    print(result)