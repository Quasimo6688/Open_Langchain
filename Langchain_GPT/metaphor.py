from metaphor_python import Metaphor
from typing import List, Dict

# 从文件中读取 API 密钥
with open('Metaphor_Key.TXT', 'r') as f:
    api_key = f.read().strip()

# 初始化 Metaphor 客户端
client = Metaphor(api_key=api_key)

# 定义搜索函数
def search(query: str) -> List[str]:
    search_results = client.search(query)
    ids = [result['id'] for result in search_results['results']]
    return ids

# 定义获取内容函数
def get_contents(ids: List[str]) -> List[Dict[str, str]]:
    contents = []
    for id in ids:
        content = client.get_content(id)
        contents.append({"id": id, "content": content})
    return contents

# 输入和输出变量
input_query = None  # 用于存储从主程序或代理接收的查询
output_contents = None  # 用于存储搜索结果，以便主程序或代理调用

# 主程序
if __name__ == "__main__":
    input_query = "Langchain技术"  # 这里模拟从主程序或代理接收到的查询
    ids = search(input_query)
    output_contents = get_contents(ids)
    print("搜索结果内容：", output_contents)
