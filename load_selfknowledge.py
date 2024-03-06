

from transformers import BertTokenizer, BertModel
import torch
import json
def extract_texts_from_data_final(data):
    texts = []
    def extract_texts(item):
        if isinstance(item, dict):
            for key, value in item.items():
                if key == 'en' and isinstance(value, str):  # Ensure it's a string in English
                    texts.append(value)
                else:
                    extract_texts(value)  # Recursive call for nested dictionaries
        elif isinstance(item, list):
            for subitem in item:
                extract_texts(subitem)  # Recursive call for items in lists

    for key, entries in data.items():
        extract_texts(entries)  # Start extraction process

    return texts
    
def simple_search_with_context_bert(query_text, context_range=1):
    # BERT模型和tokenizer的初始化应该移到这个函数外面进行
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 向量化查询文本
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).numpy()

    # 读取和处理JSON数据
    file_path = "knowledge/bonecancer/SubBoneCancer.json"  # 请替换为实际路径
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    extracted_texts = extract_texts_from_data_final(data)

    # 创建简单的向量化表示（这里使用文本长度，但可以替换为BERT向量）
    text_vectors = [len(text) for text in extracted_texts]

    # 执行搜索，找到最接近的文本及其上下文
    closest_index = min(range(len(text_vectors)), key=lambda i: abs(text_vectors[i] - len(query_text)))
    start_index = max(0, closest_index - context_range)
    end_index = min(len(extracted_texts) - 1, closest_index + context_range)
    context_texts = extracted_texts[start_index:end_index+1]

    return context_texts





if __name__ == "__main__":
    query_text = "What is the treatment for atypical cartilaginous tumors?"
    context_texts = simple_search_with_context_bert(query_text, context_range=1)
    print("Context texts found:")
    for text in context_texts:
        print(text)
