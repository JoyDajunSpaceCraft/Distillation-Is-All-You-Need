import json
import pandas as pd

df = pd.read_csv("/home/yuj49/DIAYN/data/csnlp/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv")
print(len(df))
df_label = df["Error Flag"].values.tolist()
file = "/home/yuj49/DIAYN/data/csnlp/csnlp_val.jsonl"
with open(file, "r")as f:
    data = []
    idx = 0
    
    for i in f.readlines():
        item = json.loads(i)
        item["label"] = df_label[idx]
        data.append(item)
        idx+=1
    print(idx)
store_file = "/home/yuj49/DIAYN/data/csnlp/csnlp_val1.jsonl"
with open(store_file, "w") as f:
    for i in data:
        i = json.dumps(i)
        f.write(i+"\n")


    
