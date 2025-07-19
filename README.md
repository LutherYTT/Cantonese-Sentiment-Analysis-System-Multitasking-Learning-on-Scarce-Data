# Cantonese Sentiment Analysis System: Multitasking Learning on Scarce Data

## Project Overview

This project focuses on sentiment analysis of Cantonese forum text data, performing both sentiment classification (positive, negative, neutral) and regression tasks for sentiment intensity (rating) and four emotional dimensions (delight, anger, sorrow, happiness). Built as a multi-task learning (MTL) framework, it leverages advanced NLP techniques to address the unique challenges of Cantonese text in a resource-constrained environment (~5500 labeled samples). A live demo is available via [Hugging Face Space with Gradio](#).

## Interactive Demo
The project provides an interactive demo using [Hugging Face Spaces](https://huggingface.co/spaces/your-space) with Gradio, allowing users to input text and receive sentiment predictions.

## Task Difficulty and Data Constraints

### 1. High Task Complexity
- **Multi-Task Learning (MTL)**: This project simultaneously performs:
  - **Classification**: 3-class sentiment prediction (positive, negative, neutral).
  - **Regression**: Predicting sentiment intensity (rating) and four emotional dimensions (delight, anger, sorrow, happiness).
- MTL requires balancing shared representations across tasks with differing objectives (categorical vs. continuous outputs), loss functions, and optimization strategies, making it inherently challenging.

### 2. Data Scarcity
- The dataset comprises only ~5500 labeled samples, a small scale for deep learning models like Transformers, increasing the risk of overfitting and limiting generalization.

### 3. Cantonese Specificity
- Cantonese differs significantly from Mandarin in vocabulary, grammar, sentence structure, and emotional expression. While the base model (`hfl/chinese-roberta-wwm-ext`) is a powerful Chinese pre-trained model, it was primarily trained on Mandarin corpora, potentially misaligning with Cantonese linguistic nuances.

### 4. Annotation Subjectivity
- Sentiment and emotional intensity annotations are subjective, introducing noise and inconsistency that complicates model training.

## Data Processing

To ensure high-quality input data for the model, the following meticulous preprocessing steps were executed:

### 1. Encoding Conversion
- Converted all text to **UTF-8 encoding** to correctly handle:
  - Traditional Chinese characters.
  - Cantonese-specific characters (e.g., 「咁」, 「撚」, 「俾」).
  - Emojis (e.g., ✋🏻).
- This step ensures lossless storage of data in JSON format, preventing character corruption.

### 2. Repairing Broken Text
- Identified and repaired or removed incomplete or corrupted text entries to maintain data integrity.

### 3. Low-Information Filtering
- Filtered out texts with too few characters or lacking sentiment indicators (e.g., samples with fewer than 3 characters and zero emotional scores), focusing on meaningful data.

### 4. Preserving Cantonese Expressions
- Retained unique Cantonese expressions crucial for accurate sentiment analysis.

### 5. Preserving Slang and Homophonic Puns
- Kept slang and homophonic puns that carry significant emotional value, enhancing model performance.

### 6. Additional Cleaning
- Removed duplicates, invalid entries, or non-Cantonese samples.
- Corrected sentiment label contradictions based on emotional dimension scores (e.g., marking as positive if delight + happiness > anger + sorrow).

## Technical Contributions and Solutions

### 1. Model Architecture
- **Transformer-Based MTL Model**: Built on `hfl/chinese-roberta-wwm-ext`, with a shared encoder feeding into:
  - A classification head for 3-class sentiment prediction.
  - Regression heads for sentiment intensity and four emotional dimensions.
- **Model Comparisons**: Evaluated multiple approaches:
  - SVM with TF-IDF features (baseline).
  - BERT Chinese Base.
  - `chinese-roberta-wwm-ext`.
  - `chinese-roberta-wwm-ext` with adversarial training.
  - `chinese-roberta-wwm-ext-large` with adversarial training.
  - `robert-distilled-model` (distilled from `chinese-roberta-wwm-ext-large` with adversarial training).

### 2. Knowledge Distillation
- Used `chinese-roberta-wwm-ext-large (with adversarial training)` as the teacher model to distill a smaller, more efficient `robert-distilled-model`, improving generalization under data scarcity.

### 3. Data Augmentation
- Experimented with synonym replacement and back-translation techniques to expand the dataset and mitigate the impact of limited data.

### 4. Experiments and Tuning
- **Loss Function**: Designed a weighted loss combining categorical cross-entropy (classification) and mean squared error (regression), balancing task importance.
- **Hyperparameter Tuning**: Optimized learning rate, batch size, and regularization via grid search and cross-validation.
- **Adversarial Training**: Applied to select models to enhance robustness against noisy data and domain-specific challenges.

## Models and Comparisons

### Comparison Results


## Evaluation Results

- **Best Model**: `chinese-roberta-wwm-ext (with adversarial training)` outperformed others across tasks.
- **Sentiment Classification**:
  - Accuracy: 0.933
  - Macro F1 Score: 0.923
  - Weighted F11: 0.932
- **Regression Tasks** (Mean Squared Error, MSE):
  - | **Regression** | **MSE** | **MAE** | **R2** | **Pearson r** | **Spearman rho** |
  |--|--|--|--|--|--|
  | Rating | 0.132 | 0.258 | 0.864 | 0.932 | 0.911 |
  | Delight | 0.192 | 0.181 | 0.809 | 0.903 | 0.691 |
  | Anger | 0.266 | 0.250 | 0.892 | 0.945 | 0.867 |
  | Sorrow | 0.221 | 0.216 | 0.811 | 0.907 | 0.680 |
  | Happiness | 0.261 | 0.212 | 0.878 | 0.938 | 0.797 |

These metrics highlight the Transformer-based model's superiority over traditional methods, despite data constraints.

## Challenges and Solutions

### 1. Key Challenges
- **Small Dataset**: Limited samples increased the risk of overfitting.
- **Cantonese Specificity**: Pre-trained models struggled with Cantonese nuances.
- **Multi-Task Conflicts**: Classification and regression tasks competed for shared representations.
- **Annotation Noise**: Subjective labels introduced inconsistencies.

### 2. Possible Solutions
- **Data Augmentation**: Applied synonym replacement and back-translation to increase sample diversity.
- **Loss Adjustment**: Implemented a weighted loss to dynamically prioritize tasks.
- **Model Design**: Used a shared encoder with task-specific heads to balance learning.
- **Knowledge Distillation**: Leveraged a large model's knowledge to enhance a smaller, efficient one.

### 3. Lessons Learned
- Gained expertise in designing and optimizing MTL systems.
- Developed strategies for NLP under data scarcity (augmentation, distillation).
- Learned to address domain-specific challenges (Cantonese NLP).
- Executed a full ML pipeline: data preprocessing, modeling, training, and evaluation.

## Future Directions
- **Larger Dataset**: Collect more data to explore Cantonese-specific pre-training or fine-tuning.
- **Back-Translation Augmentation**: Generate additional samples via translation to/from Mandarin/English.
- **Advanced MTL Architectures**: Experiment with dynamic task weighting or hierarchical models.
- **Semi-Supervised Learning**: Incorporate unlabeled Cantonese data to boost performance.

## Dependencies
Required libraries include:
- `transformers`
- `torch`
- `scikit-learn`
- `gradio`
- Others listed in `requirements.txt`.

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/) for providing pre-trained models.
- [Gradio](https://gradio.app/) for supporting the interactive demo.
