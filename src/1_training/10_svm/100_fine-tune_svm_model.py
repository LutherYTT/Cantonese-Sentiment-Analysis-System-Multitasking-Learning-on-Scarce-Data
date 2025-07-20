import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
import numpy as np
import joblib

# Load tokenizer
model_name = "hfl/chinese-roberta-wwm-ext"

# Read JSON
with open('../../training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Data preprocessing
texts = df['text'].tolist()
labels = df[['sentiment', 'rating', 'delight', 'anger', 'sorrow', 'happiness']].to_dict('records')

# Label encoding
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
for label in labels:
    label['sentiment'] = sentiment_map[label['sentiment']]

# Split into training set and test set
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_sentiment = [label['sentiment'] for label in train_labels]
test_sentiment = [label['sentiment'] for label in test_labels]

# Regression task label
regression_keys = ['rating', 'delight', 'anger', 'sorrow', 'happiness']
train_regression = np.array([[label[k] for k in regression_keys] for label in train_labels])
test_regression = np.array([[label[k] for k in regression_keys] for label in test_labels])

# Create feature extractor (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=20000,  
    ngram_range=(1, 2),   
    sublinear_tf=True   
)

# Convert text data
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print(f"Feature matrix shape: training set {X_train.shape}, test set {X_test.shape}")

# Training an emotion classification model (SVM)
print("Training an emotion classification model...")
sentiment_clf = SVC(
    kernel='linear',
    C=1.0,
    class_weight='balanced',
    probability=False,
    random_state=42
)
sentiment_clf.fit(X_train, train_sentiment)

# Train a regression model (using LinearSVR for multi-output regression)
print("训练回归模型...")
regressor = MultiOutputRegressor(
    LinearSVR(
        C=0.5,
        epsilon=0.1,
        loss='squared_epsilon_insensitive',
        max_iter=5000,
        random_state=42
    ),
    n_jobs=-1 
)
regressor.fit(X_train, train_regression)

# Evaluating Emotion Classification Models
print("\nEvaluating Emotion Classification Models:")
sentiment_pred = sentiment_clf.predict(X_test)

accuracy = accuracy_score(test_sentiment, sentiment_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_sentiment, sentiment_pred, average='macro', zero_division=0
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Evaluate regression models
print("\nEvaluate regression models:")
regression_pred = regressor.predict(X_test)

mse_list = []
mae_list = []
for i, key in enumerate(regression_keys):
    y_true = test_regression[:, i]
    y_pred = regression_pred[:, i]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mse_list.append(mse)
    mae_list.append(mae)

    print(f"\n{key} Regression result:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

# Overall regression metrics
print("Overall regression metrics:")
print(f"Average MSE: {np.mean(mse_list):.4f}")
print(f"Average MAE: {np.mean(mae_list):.4f}")

# 保存模型
print("\nSaving Model...")
joblib.dump(vectorizer, './svm/svm_vectorizer.pkl')
joblib.dump(sentiment_clf, './svm/sentiment_clf.pkl')
joblib.dump(regressor, './svm/regressor.pkl')
print("Model saved successfully!")