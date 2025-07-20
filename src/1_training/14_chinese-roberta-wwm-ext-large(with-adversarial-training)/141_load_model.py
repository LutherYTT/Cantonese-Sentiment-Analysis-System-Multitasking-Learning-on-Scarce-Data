import torch
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoConfig
import torch.nn as nn

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
        pooled_output = outputs[1]  # Use pooled output (CLS token)

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

# Load the saved model
model = MultiTaskBert.from_pretrained("./content/best_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    model.eval()
    with torch.no_grad():
        sentiment_logits, regression_outputs = model(**inputs)

    # Process outputs
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
    sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
    print(f"Sentiment: {sentiment_map[sentiment_pred]}")
    print(f"Rating: {regression_outputs[0, 0].item():.2f}")
    print(f"Delight: {regression_outputs[0, 1].item():.2f}")
    print(f"Anger: {regression_outputs[0, 2].item():.2f}")
    print(f"Sorrow: {regression_outputs[0, 3].item():.2f}")
    print(f"Happiness: {regression_outputs[0, 4].item():.2f}")

    return regression_outputs

# if __name__ == "__main__":
#     text = "呢個世界真係好美好"
#     predict(text)