import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

import pandas as pd
import json
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, AutoModel, AutoTokenizer,AutoConfig, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "hfl/chinese-roberta-wwm-ext-large"
config = AutoConfig.from_pretrained(model_name)

class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained(model_name, config=config)  # Matches training definition
        self.classifier = nn.Linear(config.hidden_size, 3)  # 3 classes for sentiment
        self.regressor = nn.Linear(config.hidden_size, 5)   # 5 regression outputs

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load teach model and tokenizer
teacher_model = MultiTaskBert.from_pretrained("./content/best_model").to(device)

teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# Read JSON
with open('../../training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)
#df = df.head(500)

# Data preprocess
texts = df['text'].tolist()
labels = df[['sentiment', 'rating', 'delight', 'anger', 'sorrow', 'happiness']].to_dict('records')

# Label encoding
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
for label in labels:
    # Convert sentiment label to lowercase before mapping
    sentiment_key = label['sentiment'].lower()
    if sentiment_key in sentiment_map:
        label['sentiment'] = sentiment_map[sentiment_key]
    else:
        # Handle unknown labels (fallback to neutral)
        label['sentiment'] = sentiment_map['neutral']

# Ensure all sentiment labels are in [0, 2]
for label in labels:
    assert label['sentiment'] in [0, 1, 2], f"Invalid sentiment label: {label['sentiment']}"

# Split into training set and test set
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

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

# Uses Chinee RoBERTa's tokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# Load RoBERTa model
student_base_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

from transformers import Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error

def compute_metrics(eval_pred):
    logits, regression_outputs = eval_pred.predictions
    labels = eval_pred.label_ids

    # Clasification task metrics
    preds = np.argmax(logits, axis=1)
    true = labels['sentiment']
    accuracy = accuracy_score(true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, preds, average='macro', zero_division=0
    )

    # Regression task metrics
    mse_list = []
    mae_list = []
    for i, key in enumerate(['rating', 'delight', 'anger', 'sorrow', 'happiness']):
        y_true = labels[key]
        y_pred = regression_outputs[:, i]
        mse_list.append(mean_squared_error(y_true, y_pred))
        mae_list.append(mean_absolute_error(y_true, y_pred))

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

def distillation_loss(teacher_outputs, student_outputs, labels,
                      alpha=0.5, temperature=3.0):
    # — unpack teacher outputs —
    if isinstance(teacher_outputs, tuple):
        # assume (sentiment_logits, regression_outputs)
        t_sent_logits, t_reg_outputs = teacher_outputs
    else:
        t_sent_logits = teacher_outputs['sentiment_logits']
        t_reg_outputs = teacher_outputs['regression_outputs']

    # classification loss (student vs. gold)
    sentiment_loss = nn.CrossEntropyLoss()(
        student_outputs['logits'], labels['sentiment']
    )

    # regression loss (student vs. gold)
    regression_targets = torch.stack([
        labels['rating'],
        labels['delight'],
        labels['anger'],
        labels['sorrow'],
        labels['happiness']
    ], dim=1)
    regression_loss = nn.MSELoss()(
        student_outputs['regression_outputs'], regression_targets
    )

    # distillation on the soft‐labels
    teacher_probs = torch.softmax(t_sent_logits / temperature, dim=-1)
    student_log_probs = torch.log_softmax(
        student_outputs['logits'] / temperature, dim=-1
    )
    distill_loss = nn.KLDivLoss(reduction='batchmean')(
        student_log_probs, teacher_probs
    ) * (temperature ** 2)

    # combine
    total_loss = (
        alpha * distill_loss
        + (1 - alpha) * sentiment_loss
        + regression_loss
    )
    return total_loss

class MultiTaskRoberta(nn.Module):  
    def __init__(self, base_model):
        super().__init__()
        self.roberta = base_model 
        self.classifier = nn.Linear(768, 3)  # Claififcation
        self.regressor = nn.Linear(768, 5)   # Regreion

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0] 
        logits = self.classifier(pooled_output)
        regression_outputs = self.regressor(pooled_output)

        if labels is not None:
            # Ensure teacher models are on the same device
            teacher_device = next(teacher_model.parameters()).device
            teacher_input_ids = input_ids.to(teacher_device)
            teacher_attention_mask = attention_mask.to(teacher_device) if attention_mask is not None else None

            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_input_ids, attention_mask=teacher_attention_mask)

            loss = distillation_loss(
                teacher_outputs,
                {'logits': logits, 'regression_outputs': regression_outputs},
                labels
            )
            return {"loss": loss, "logits": logits, "regression_outputs": regression_outputs}
        return {"logits": logits, "regression_outputs": regression_outputs}

# Initialise student model and move it to the device
student_model = MultiTaskRoberta(student_base_model).to(device)

print(f"Teacher model device: {next(teacher_model.parameters()).device}")
print(f"Student model device: {next(student_model.parameters()).device}")

import wandb
wandb.login(key="YOUR API KEY")

# Training parameters
training_args = TrainingArguments(
    output_dir="./distilled_model_temp",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),  
    no_cuda=not torch.cuda.is_available(), 
    optim="adafactor",
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./robert-distilled-model")