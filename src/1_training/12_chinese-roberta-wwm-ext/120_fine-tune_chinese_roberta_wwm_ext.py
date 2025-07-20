import pandas as pd
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 加载 tokenizer 和 config
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config    = AutoConfig.from_pretrained(model_name)

# 讀取 JSON 數據
with open('../../training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
#df = df.head(500)

# 數據預處理
texts = df['text'].tolist()
labels = df[['sentiment', 'rating', 'delight', 'anger', 'sorrow', 'happiness']].to_dict('records')

# 標籤編碼
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
for label in labels:
    label['sentiment'] = sentiment_map[label['sentiment']]
    # rating 和情緒強度已是數值，無需轉換

# 分訓練集和測試集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 用 tokenizer 將文本轉成 token IDs
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# 自定義數據集類
class CantoneseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = self.labels[idx]
        item['labels'] = {
            'sentiment': torch.tensor(label['sentiment']),
            'rating': torch.tensor(label['rating'], dtype=torch.float),
            'delight': torch.tensor(label['delight'], dtype=torch.float),
            'anger': torch.tensor(label['anger'], dtype=torch.float),
            'sorrow': torch.tensor(label['sorrow'], dtype=torch.float),
            'happiness': torch.tensor(label['happiness'], dtype=torch.float)
        }
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CantoneseDataset(train_encodings, train_labels)
test_dataset = CantoneseDataset(test_encodings, test_labels)

from transformers import Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error

def compute_metrics(eval_pred):
    logits, regression_outputs = eval_pred.predictions
    labels = eval_pred.label_ids

    # --- 分类任务指标 ---
    preds = np.argmax(logits, axis=1)
    true = labels['sentiment']
    accuracy = accuracy_score(true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, preds, average='macro', zero_division=0
    )

    # --- 回归任务指标 ---
    # 对每个维度计算 MSE 和 MAE，然后求平均
    mse_list = []
    mae_list = []
    for i, key in enumerate(['rating', 'delight', 'anger', 'sorrow', 'happiness']):
        y_true = labels[key]
        y_pred = regression_outputs[:, i]
        mse_list.append(mean_squared_error(y_true, y_pred))
        mae_list.append(mean_absolute_error(y_true, y_pred))

    # 聚合回归指标（也可按需单独汇报）
    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))

    return {
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'mae': mae,
    }

# Define the custom data collator
def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key != 'labels':
            batch[key] = torch.stack([f[key] for f in features])
    labels = {}
    for label_key in features[0]['labels'].keys():
        labels[label_key] = torch.stack([f['labels'][label_key] for f in features])
    batch['labels'] = labels
    return batch

# 自定義多任務模型
class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        from transformers import RobertaModel
        self.bert = RobertaModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.regressor = nn.Linear(config.hidden_size, 5)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        sentiment_logits = self.classifier(pooled_output)
        regression_outputs = self.regressor(pooled_output)

        if labels is not None:
            sentiment_labels = labels['sentiment']
            rating_labels = labels['rating']
            delight_labels = labels['delight']
            anger_labels = labels['anger']
            sorrow_labels = labels['sorrow']
            happiness_labels = labels['happiness']

            loss_fct_cls = nn.CrossEntropyLoss()
            loss_fct_reg = nn.MSELoss()

            sentiment_loss = loss_fct_cls(sentiment_logits, sentiment_labels)
            rating_loss = loss_fct_reg(regression_outputs[:, 0], rating_labels)
            delight_loss = loss_fct_reg(regression_outputs[:, 1], delight_labels)
            anger_loss = loss_fct_reg(regression_outputs[:, 2], anger_labels)
            sorrow_loss = loss_fct_reg(regression_outputs[:, 3], sorrow_labels)
            happiness_loss = loss_fct_reg(regression_outputs[:, 4], happiness_labels)

            total_loss = sentiment_loss + rating_loss + delight_loss + anger_loss + sorrow_loss + happiness_loss
            return {"loss": total_loss, "sentiment_logits": sentiment_logits, "regression_outputs": regression_outputs}
        else:
            return sentiment_logits, regression_outputs

# Load model
model = MultiTaskBert.from_pretrained(model_name, config=config)

# 設置訓練參數
training_args = TrainingArguments(
    output_dir="./cantonese_sentiment",
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_ratio=0.25,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    fp16=True,
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
)

import wandb
wandb.login(key="Use Your own API Key")

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./best_model")