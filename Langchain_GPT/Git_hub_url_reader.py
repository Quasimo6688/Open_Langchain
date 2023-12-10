import requests
import json
import time
import os


def fetch_image_urls_github_api(repo, branch, path, token, output_dir, rate_limit=1):
    print("开始获取GitHub仓库中的图片URL...")
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print("获取数据失败:", response.json())
        return

    image_urls = {}
    for file in response.json():
        if file['name'].endswith('.png'):
            image_urls[file['name']] = file['download_url']
            print(f"已获取图片URL: {file['name']}")
            time.sleep(rate_limit)  # 等待指定的秒数以避免超过API速率限制

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    output_file = os.path.join(output_dir, 'URL_Pictures_map.json')
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(image_urls, file, indent=4, ensure_ascii=False)

    print(f"图片URL已保存到 {output_file}")




#运行Github抓取程序
repo = 'Quasimo6688/Open_Langchain'  # 格式为 '用户名/仓库名'
branch = 'cf86ca737a1c7f9494922d93b9fd4fbba3f22361'  # 分支的commit hash
path = 'Langchain_GPT/Embedding_Files/Pictures'  # 仓库中的文件夹路径
token = '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'  # 替换为您的GitHub个人访问令牌
output_dir = 'Embedding_Files'  # JSON文件的保存目录
fetch_image_urls_github_api(repo, branch, path, token, output_dir)
