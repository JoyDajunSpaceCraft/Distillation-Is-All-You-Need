import torch
from torch.utils.data import Dataset
import json

class MedicalDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        full_text = entry["full_text"]
        knowledge = entry["knowledge"]
        reason = entry["reason"]
        label = entry["label"]

        # 将 full_text 和 knowledge 连接作为输入
        inputs = self.tokenizer(f"{full_text} [SEP] {knowledge}", max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 为生成任务编码 reason
        reason_ids = self.tokenizer(reason, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")['input_ids'].squeeze(0)

        return input_ids, attention_mask, reason_ids, torch.tensor(label)

def evaluate_model(dataloader, model):
    model.eval()  # 确保模型处于评估模式
    predictions, true_labels = [], []
    generated_texts, reference_texts = []

    with torch.no_grad():
        for input_ids, attention_mask, reason_ids, labels in dataloader:
            # 分类评估
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 0, :]
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

            # 生成任务评估
            generated = generator(input_ids, max_length=150)
            generated_texts.extend([g['generated_text'] for g in generated])
            reference_texts.extend([tokenizer.decode(reason_ids[i], skip_special_tokens=True) for i in range(len(reason_ids))])

    # 计算F1得分
    f1 = f1_score(true_labels, predictions, average='binary')

    # 计算ROUGE得分
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
    rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

    # 在评估结束后不应将模型设置回训练模式
    return f1, rouge1, rouge2, rougeL


from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW

# 初始化分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 加载数据集
dataset = MedicalDataset('/home/yuj49/DIAYN/data/csnlp/csnlp_train.jsonl', tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)


for epoch in range(3):  # 训练3个epochs
    for input_ids, attention_mask, reason_ids, labels in dataloader:
        model.train()
        # 前向传播，获取模型输出
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=reason_ids)

        # 生成任务的损失
        loss = outputs.loss

        # 对于分类任务，我们需要从模型输出中提取特定的向量来进行分类
        # 假设我们用最后一个隐藏状态的第一个向量作为分类任务的输入
        logits = outputs.logits[:, 0, :]  # 假设分类头部已经在模型中
        classification_loss_fn = CrossEntropyLoss()
        classification_loss = classification_loss_fn(logits, labels)

        # 总损失是生成损失和分类损失的和
        total_loss = loss + classification_loss

        # 后向传播和优化
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {total_loss.item()}")
    f1, rouge1, rouge2, rougeL = evaluate_model(dataloader, model)
    print(f"Epoch {epoch}, F1 Score: {f1}, ROUGE-1: {rouge1}, ROUGE-2: {rouge2}, ROUGE-L: {rougeL}")