# coding=utf-8
from langflow import load_flow_from_json

flow = load_flow_from_json("./data/Langflow_test.json")
# 现在你可以像使用任何其他链一样使用它
result = flow("给我普及下飞行知识并出一些问答题")
print(result['response'])