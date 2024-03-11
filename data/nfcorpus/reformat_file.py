# import json
# import re
# reason = "reason_bm25_results.json"
# original = "refined_bm25_docs.json"
# gen_file = "nfcorpus_100_reason.jsonl"
# with open(reason, "r")as f1, open(original,"r") as f2:
#     reason_file = json.load(f1)
#     original_file = json.load(f2)
# def clean_reason(permutation):
#     # 找到第一次出现的排名序列
#     ranking_pattern = re.findall(r'\[\d+\] >', permutation)
#     new_response = ""
#     # 提取这些模式中的数字
#     ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
#     new_response = " ".join([str(i )for i in ranking_numbers])
#     # 单独处理最后一个排名元素
#     # 查找文本中的最后一个 [数字] 模式
#     last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', permutation)
#     if last_number_match:
#         last_number = last_number_match.group().strip('[]')
#         new_response +=  " " + last_number
#         # ranking_numbers.append(str(last_number))
#     # print("new_response",new_response)
#     return new_response

# with open(gen_file, "w") as f:

#     for reason, original in zip(reason_file[:2], original_file[:2]):
#         # query sorted_docids  re_rank_id  sorted_docs unsorted_docs  reason
#         new_item ={}
#         new_item["reason"] = reason["reason"]
#         new_item["sorted_docids"] = reason["sorted_docids"]
#         new_item["query"] = reason["query"]
#         ids = clean_reason(new_item["reason"])
#         ids = ids.split()
#         new_item["re_rank_id"]=ids
#         new_item["sorted_docs"] = []
#         new_item["unsorted_docs"] = []
#         new_item["sorted_score"] = []
#         for hit in original["hits"]:
#             new_item["unsorted_docs"].append(hit["content"])
#         for idx in  new_item["sorted_docids"]:
#             for hit in original["hits"]:
#                 if hit["docid"] == idx:
#                     new_item["sorted_docs"].append(hit["content"])
#                     new_item["sorted_score"].append(hit["score"])
#         print(new_item)
#         new_json = json.dumps(new_item)
#         f1.write(new_json)
        

import json
import re

# 文件路径可能需要根据你的文件存储位置进行调整
reason_path = "reason_bm25_results.json"
original_path = "refined_bm25_docs.json"
gen_file_path = "nfcorpus_100_reason.jsonl"

# 加载JSON文件
with open(reason_path, "r") as f1, open(original_path, "r") as f2:
    reason_file = json.load(f1)
    original_file = json.load(f2)

# 定义清理理由文本的函数
def clean_reason(permutation):
    ranking_pattern = re.findall(r'\[\d+\] >', permutation)
    new_response = ""
    ranking_numbers = [int(num) for pattern in ranking_pattern for num in re.findall(r'\d+', pattern)]
    new_response = " ".join([str(i) for i in ranking_numbers])
    last_number_match = re.search(r'\[\d+\](?!.*\[\d+\])', permutation)
    if last_number_match:
        last_number = last_number_match.group().strip('[]')
        new_response += " " + last_number
    return new_response

# 生成新的JSONL文件
with open(gen_file_path, "w") as f:
    for reason, original in zip(reason_file, original_file):
        new_item = {
            "reason": reason["reason"],
            "sorted_docids": reason["sorted_docids"],
            "query": reason["query"],
            "re_rank_id": clean_reason(reason["reason"]).split(),
            "sorted_docs": [],
            "unsorted_docs": [],
            "sorted_score": []
        }

        for hit in original["hits"]:
            new_item["unsorted_docs"].append(hit["content"])
            if hit["docid"] in new_item["sorted_docids"]:
                new_item["sorted_docs"].append(hit["content"])
                new_item["sorted_score"].append(hit["score"])
                
        # 将新的JSON对象写入文件并添加换行符
        f.write(json.dumps(new_item) + '\n')
