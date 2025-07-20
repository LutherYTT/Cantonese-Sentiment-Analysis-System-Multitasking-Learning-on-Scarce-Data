import torch
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
import torch.nn as nn

class MultiTaskBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.regressor = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]

        sentiment_logits = self.classifier(pooled_output)
        regression_outputs = self.regressor(pooled_output)

        return sentiment_logits, regression_outputs

# Path to your checkpoint folder
checkpoint_path = "./best_model" 

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = MultiTaskBert.from_pretrained(checkpoint_path)

# Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        sentiment_logits, regression_outputs = model(**inputs)

    # Process outputs
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    rating_pred = regression_outputs[0, 0].item()
    delight_pred = regression_outputs[0, 1].item()
    anger_pred = regression_outputs[0, 2].item()
    sorrow_pred = regression_outputs[0, 3].item()
    happiness_pred = regression_outputs[0, 4].item()

    sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    return {
        'sentiment': sentiment_map[sentiment_pred],
        'rating': rating_pred,
        'delight': delight_pred,
        'anger': anger_pred,
        'sorrow': sorrow_pred,
        'happiness': happiness_pred
    }

# if __name__ == "__main__":
#     text = "呢個世界真係好美好"
#     predict(text)