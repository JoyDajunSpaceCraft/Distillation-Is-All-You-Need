import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加载分词器和模型
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 假设你的数据存储在一个名为 "training_data.json" 的文件中
data_file = 'data/nfcorpus/nf.jsonl'

def load_and_process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    queries = data['query']
    docs = data['unsorted_docs']
    reasons = data['reason']

    # 将查询和文档合并为一个输入字符串
    input_texts = []
    for query, doc in zip(queries, docs):
        input_text = f"Query: {query} Document: {' '.join(doc)}"
        input_texts.append(input_text)

    # 生成的输出是理由
    output_texts = [reasons]

    # 编码文本
    input_encodings = tokenizer(input_texts, truncation=True, padding=True, max_length=512)
    output_encodings = tokenizer(output_texts, truncation=True, padding=True, max_length=512)

    return input_encodings, output_encodings

# 加载和处理数据
input_encodings, output_encodings = load_and_process_data(data_file)

# 自定义数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_encodings, output_encodings):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings

    def __getitem__(self, idx):
        input_item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        output_item = {key: torch.tensor(val[idx]) for key, val in self.output_encodings.items()}
        return input_item, output_item

    def __len__(self):
        return len(self.input_encodings['input_ids'])

# 创建数据集实例
dataset = MyDataset(input_encodings, output_encodings)

# 训练参数配置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 开始微调
trainer.train()

# 保存模型
model.save_pretrained("./finetuned_model")
