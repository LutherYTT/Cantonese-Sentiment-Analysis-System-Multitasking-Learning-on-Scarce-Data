import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Seaborn style
sns.set(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '0.15'
plt.rcParams['axes.linewidth'] = 1.2

# Define model name list
models = ['svm', 'bert-chinese-base', 'chinese-roberta-wwm-ext', 
          'chinese-roberta-wwm-ext(with-adversarial-training)', 
          'chinese-roberta-wwm-ext-large(with-adversarial-training)', 
          'robert-distilled-model']

# Load all JSON data
data = {}
for model in models:
    try:
        filename = f"../../assets/{model}/evaluation_results.json"
        with open(filename, 'r') as f:
            data[model] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found for model {model}")
        data[model] = {'sentiment': {}, 'regression': {}}

# Format model name
model_name = [
    'SVM', 
    'BERT\nBase', 
    'RoBERTa\nBase', 
    'RoBERTa\n(AT)', 
    'RoBERTa-Large\n(AT)', 
    'Distilled\nRoBERTa'
]

# sentiment classification summary chart
# Collect all emotional classification metrics
sentiment_metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                     'weighted_precision', 'weighted_recall', 'weighted_f1']

sentiment_data = []
for model in models:
    model_data = data[model].get('sentiment', {})
    for metric in sentiment_metrics:
        value = model_data.get(metric, 0)
        sentiment_data.append({
            'Model': model_name[models.index(model)],
            'Metric': metric.replace('_', ' ').title(),
            'Value': value
        })

sentiment_df = pd.DataFrame(sentiment_data)

# Create summarychart
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
fig.suptitle('Sentiment Classification Performance Comparison', fontsize=20, fontweight='bold')

# Set a uniform colour mapping
palette = sns.color_palette("viridis", len(model_name))

for i, metric in enumerate(sentiment_metrics):
    ax = axes[i]
    metric_name = metric.replace('_', ' ').title()
    metric_df = sentiment_df[sentiment_df['Metric'] == metric_name]
    
    sns.barplot(x='Model', y='Value', data=metric_df, ax=ax, 
                palette=palette, edgecolor='black', linewidth=1)
    
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=20, labelsize=10)
    
    # Add numerical labels
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=9)

# Remove the last empty subgraph
fig.delaxes(axes[-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../../assets/evaluation/sentiment_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Regression Task Summary Chart
# Define regression metrics
regression_metrics = ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_rho']
metric_names = {
    'rmse': 'RMSE',
    'mae': 'MAE',
    'r2': 'R²',
    'pearson_r': "Pearson's r",
    'spearman_rho': "Spearman's ρ"
}

# Define regression task objectives
regression_targets = ['rating', 'delight', 'anger', 'sorrow', 'happiness']

# Create a summary chart for each regression task
for target in regression_targets:
    # Collect all metrics for the current task.
    task_data = []
    for model in models:
        model_data = data[model].get('regression', {}).get(target, {})
        for metric in regression_metrics:
            value = model_data.get(metric, 0)
            task_data.append({
                'Model': model_name[models.index(model)],
                'Metric': metric_names[metric],
                'Value': value
            })
    
    task_df = pd.DataFrame(task_data)
    
    # Create summary chart for regression tasks
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{target.capitalize()} Regression Task Performance Comparison', 
                fontsize=20, fontweight='bold')
    
    # Set the y-axis range
    y_limits = {
        'RMSE': (0, task_df[task_df['Metric'] == 'RMSE']['Value'].max() * 1.2),
        'MAE': (0, task_df[task_df['Metric'] == 'MAE']['Value'].max() * 1.2),
        'R²': (-0.1, 1.05),
        "Pearson's r": (-0.1, 1.05),
        "Spearman's ρ": (-0.1, 1.05)
    }
    
    # Draw sub-plot
    for i, metric_name in enumerate(metric_names.values()):
        ax = axes[i//3, i%3]
        metric_df = task_df[task_df['Metric'] == metric_name]
        
        sns.barplot(x='Model', y='Value', data=metric_df, ax=ax, 
                    palette=palette, edgecolor='black', linewidth=1)
        
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', rotation=20, labelsize=10)
        
        # setting y-axis range
        if metric_name in y_limits:
            ax.set_ylim(y_limits[metric_name])
        
        # add numerical label
        for p in ax.patches:
            height = p.get_height()
            va = 'bottom' if height < 0 else 'center'
            y_pos = height if height >= 0 else height - 0.05
            ax.annotate(f'{height:.3f}', 
                        (p.get_x() + p.get_width() / 2., y_pos),
                        ha='center', va=va, 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=10)
    
    # remove last empty graph
    if len(regression_metrics) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'../../assets/evaluation/{target}_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()