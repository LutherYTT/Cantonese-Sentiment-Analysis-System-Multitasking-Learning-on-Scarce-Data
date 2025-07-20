import torch
from transformers import AutoTokenizer

# Path to your checkpoint folder
checkpoint_path = "./best_model"

# Load tokenizer and model
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MultiTaskBert.from_pretrained(checkpoint_path, config=config)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        sentiment_logits, regression_outputs = model(**inputs)

    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    return {
        'sentiment': sentiment_map[sentiment_pred],
        'rating': regression_outputs[0, 0].item(),
        'delight': regression_outputs[0, 1].item(),
        'anger': regression_outputs[0, 2].item(),
        'sorrow': regression_outputs[0, 3].item(),
        'happiness': regression_outputs[0, 4].item()
    }

# if __name__ == "__main__":
    # result = predict("我非常喜欢這個模型！")
    # print(result)
