from langchain.tools import DuckDuckGoSearchRun

# 输入和输出变量
input_query = None  # 用于存储从主程序或代理接收的查询
output_results = None  # 用于存储搜索结果，以便主程序或代理调用

# 搜索函数
def search(query):
    global output_results
    search_tool = DuckDuckGoSearchRun()
    search_tool.args_schema = None  # 这里可以设置参数模型，如果有的话
    search_tool.description = "这是一个用于DuckDuckGo搜索的工具,当你需要回答有关时事问题时很有用"
    search_tool.name = "DuckDuckGo搜索工具"
    search_tool.return_direct = True  # 是否直接返回工具的输出
    search_tool.tags = ["搜索", "DuckDuckGo"]
    search_tool.verbose = True  # 是否记录工具的进度
    output_results = search_tool.run(query)

# 主程序
if __name__ == "__main__":
    input_query = "北京 今日 尾号 限行"  # 这里模拟从主程序或代理接收到的查询
    search(input_query)
    print("搜索结果：", output_results)
