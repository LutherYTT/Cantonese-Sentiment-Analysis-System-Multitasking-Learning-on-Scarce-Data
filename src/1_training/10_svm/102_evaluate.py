import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load SVM model and TF-IDF vectorizer
print("Load SVM model...")
vectorizer = joblib.load('./svm/svm_vectorizer.pkl')
sentiment_clf = joblib.load('./svm/sentiment_clf.pkl')
regressor = joblib.load('./svm/regressor.pkl')
print("Model loaded!")

# Load and preprocess data
with open('../../training_data', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# label encoding
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# split test set
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare test set
test_texts = test_df['text'].tolist()
test_sentiment = test_df['sentiment'].tolist()

# Regression task label
regression_keys = ['rating', 'delight', 'anger', 'sorrow', 'happiness']
test_regression = test_df[regression_keys].values

# Convert text to TF-IDF features
print("Convert text to TF-IDF features...")
X_test = vectorizer.transform(test_texts)
print(f"Test set feature matrix shape: {X_test.shape}")

def evaluate_svm():
    # Store all prediction and ground truth
    all_sentiment_preds = []
    all_sentiment_true = test_sentiment

    regression_targets = {
        'rating': [], 'delight': [], 'anger': [],
        'sorrow': [], 'happiness': []
    }
    regression_preds = {
        'rating': [], 'delight': [], 'anger': [],
        'sorrow': [], 'happiness': []
    }

    # Emotion classification prediction
    print("Emotion classification prediction...")
    sentiment_pred = sentiment_clf.predict(X_test)
    all_sentiment_preds = sentiment_pred.tolist()

    # regression prediction
    print("regression prediction...")
    regression_outputs = regressor.predict(X_test)

    # Fill in the actual values
    regression_targets['rating'] = test_regression[:, 0].tolist()
    regression_targets['delight'] = test_regression[:, 1].tolist()
    regression_targets['anger'] = test_regression[:, 2].tolist()
    regression_targets['sorrow'] = test_regression[:, 3].tolist()
    regression_targets['happiness'] = test_regression[:, 4].tolist()

    # Fill in predicted values
    regression_preds['rating'] = regression_outputs[:, 0].tolist()
    regression_preds['delight'] = regression_outputs[:, 1].tolist()
    regression_preds['anger'] = regression_outputs[:, 2].tolist()
    regression_preds['sorrow'] = regression_outputs[:, 3].tolist()
    regression_preds['happiness'] = regression_outputs[:, 4].tolist()

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

results = evaluate_svm()

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
plt.title('SVM Sentiment Classification Confusion Matrix')
plt.savefig('svm_confusion_matrix.png', bbox_inches='tight')
plt.show()
print("Confusion matrix has been saved as svm_confusion_matrix.png.")

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

with open('svm_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

print("\nSVM evaluation results have been saved as svm_evaluation_results.json.")