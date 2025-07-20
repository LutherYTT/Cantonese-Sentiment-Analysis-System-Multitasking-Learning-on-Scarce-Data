import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, AutoConfig
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split

class MultiTaskBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.regressor = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
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

# Load model and weight
model_dir = "./best_model"
config = AutoConfig.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskBert(config).to(device)

# Load safetensors weight
model_weights = load_file(f"{model_dir}/model.safetensors")
model.load_state_dict(model_weights)
model.eval()

# Load and preprocess data
with open('../../training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# label encoding
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# split test set
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        label = self.labels.iloc[idx]
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'sentiment': torch.tensor(label['sentiment'], dtype=torch.long),
            'rating': torch.tensor(label['rating'], dtype=torch.float),
            'delight': torch.tensor(label['delight'], dtype=torch.float),
            'anger': torch.tensor(label['anger'], dtype=torch.float),
            'sorrow': torch.tensor(label['sorrow'], dtype=torch.float),
            'happiness': torch.tensor(label['happiness'], dtype=torch.float)
        }

test_dataset = TestDataset(test_df['text'].tolist(), test_df)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def evaluate(model, dataloader):
    # Store all prediction and ground truth
    all_sentiment_preds = []
    all_sentiment_true = []

    regression_targets = {
        'rating': [], 'delight': [], 'anger': [],
        'sorrow': [], 'happiness': []
    }
    regression_preds = {
        'rating': [], 'delight': [], 'anger': [],
        'sorrow': [], 'happiness': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # Get ground truth
            sentiment_true = batch['sentiment'].cpu().numpy()
            all_sentiment_true.extend(sentiment_true)

            for key in regression_targets:
                regression_targets[key].extend(batch[key].cpu().numpy())

            # Model prediction
            sentiment_logits, regression_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Classification Task
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            all_sentiment_preds.extend(sentiment_preds)

            # Regression Task
            regression_outputs = regression_outputs.cpu().numpy()

            for i, key in enumerate(['rating', 'delight', 'anger', 'sorrow', 'happiness']):
                regression_preds[key].extend(regression_outputs[:, i])

    # Classification task evaluation
    sentiment_metrics = {}

    # Overall metrics
    accuracy = accuracy_score(all_sentiment_true, all_sentiment_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_sentiment_true, all_sentiment_preds, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_sentiment_true, all_sentiment_preds, average='weighted', zero_division=0
    )

    sentiment_metrics['accuracy'] = accuracy
    sentiment_metrics['macro_precision'] = precision
    sentiment_metrics['macro_recall'] = recall
    sentiment_metrics['macro_f1'] = f1
    sentiment_metrics['weighted_precision'] = weighted_precision
    sentiment_metrics['weighted_recall'] = weighted_recall
    sentiment_metrics['weighted_f1'] = weighted_f1

    # Metrics for each category
    class_report = {}
    classes = ['positive', 'negative', 'neutral']
    for i, cls in enumerate(classes):
        cls_true = [1 if t == i else 0 for t in all_sentiment_true]
        cls_pred = [1 if p == i else 0 for p in all_sentiment_preds]

        precision, recall, f1, _ = precision_recall_fscore_support(
            cls_true, cls_pred, average='binary', zero_division=0
        )
        class_report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(cls_true)
        }

    # Confusion Matrix
    cm = confusion_matrix(all_sentiment_true, all_sentiment_preds)

    # Regression task evaluation
    regression_metrics = {}
    for key in regression_targets:
        y_true = np.array(regression_targets[key])
        y_pred = np.array(regression_preds[key])

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        regression_metrics[key] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_corr,
            'spearman_rho': spearman_corr
        }

    return {
        'sentiment': sentiment_metrics,
        'class_report': class_report,
        'confusion_matrix': cm.tolist(),
        'regression': regression_metrics
    }

results = evaluate(model, test_loader)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

print("\nClassification Task Evaluation Results")
print(f"Accuracy: {results['sentiment']['accuracy']:.4f}")
print(f"Macro Precision: {results['sentiment']['macro_precision']:.4f}")
print(f"Macro Recall: {results['sentiment']['macro_recall']:.4f}")
print(f"Macro F1: {results['sentiment']['macro_f1']:.4f}")
print(f"Weighted Precision: {results['sentiment']['weighted_precision']:.4f}")
print(f"Weighted Recall: {results['sentiment']['weighted_recall']:.4f}")
print(f"Weighted F1: {results['sentiment']['weighted_f1']:.4f}")

print("\nDetailed metrics for each category:")
for cls, metrics in results['class_report'].items():
    print(f"{cls}:")
    print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | Samples: {metrics['support']}")

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    results['confusion_matrix'],
    annot=True, fmt='d',
    cmap='Blues',
    xticklabels=['positive', 'negative', 'neutral'],
    yticklabels=['positive', 'negative', 'neutral']
)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Sentiment Classification Confusion Matrix')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()
print("Confusion matrix has been saved as confusion_matrix.png")

print("\nRegression Task Evaluation Results")
print("Detailed metrics for each dimension:")
for target, metrics in results['regression'].items():
    print(f"\n{target}:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman ρ: {metrics['spearman_rho']:.4f}")

with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

print("\nEvaluation results have been saved as evaluation_results.json")