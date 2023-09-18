from googleapiclient.discovery import build
from langchain.utilities import GoogleSearchAPIWrapper
import os

# 读取API密钥和CSE ID
def read_google_credentials():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_file = os.path.join(script_dir, 'GOOGLE_API_KEY.TXT')
    cse_id_file = os.path.join(script_dir, 'GOOGLE_CSE_ID.TXT')

    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()

    with open(cse_id_file, 'r') as f:
        cse_id = f.read().strip()

    return api_key, cse_id

# 初始化谷歌搜索API
def initialize_google_search():
    api_key, cse_id = read_google_credentials()
    google_search_service = build("customsearch", "v1", developerKey=api_key)
    return google_search_service, cse_id

# 使用LangChain的GoogleSearchAPIWrapper进行搜索
def search_google_with_langchain(query, google_search_service, cse_id):
    google_search_wrapper = GoogleSearchAPIWrapper(service=google_search_service, cse_id=cse_id)
    results = google_search_wrapper.search(query)
    return results

# 示例用法
if __name__ == "__main__":
    google_search_service, cse_id = initialize_google_search()
    query = "What is the meaning of life?"
    results = search_google_with_langchain(query, google_search_service, cse_id)

    print(results)
