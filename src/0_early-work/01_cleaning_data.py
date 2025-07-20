import json
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure reproducible langdetect results
DetectorFactory.seed = 42

def clean_data(data):
    cleaned_data = []
    seen_texts = set() 

    for sample in data:
        # Handle missing values: Skip samples with missing or empty 'text' or required sentiment fields
        if 'text' not in sample or not sample['text'].strip():
            continue
        if 'sentiment' not in sample \
           or 'delight' not in sample \
           or 'happiness' not in sample \
           or 'anger' not in sample \
           or 'sorrow' not in sample:
            continue

        text = sample['text'].strip()

        # Remove duplicates: Skip if text has already been seen
        if text in seen_texts:
            continue
        seen_texts.add(text)

        # Skip pure punctuation or non-alphanumeric text
        if all(not c.isalnum() for c in text):
            continue

        # Detect language
        try:
            lang = detect(text)
        except LangDetectException:
            lang = 'unknown'

        # Remove English samples with no sentiment
        if lang == 'en' and all(sample[key] == 0 for key in ['delight', 'happiness', 'anger', 'sorrow']):
            continue

        # Remove short samples (<3 chars) with no sentiment
        if len(text) < 3 and all(sample[key] == 0 for key in ['delight', 'happiness', 'anger', 'sorrow']):
            continue

        # Fix sentiment label contradictions
        pos_score = sample['delight'] + sample['happiness']
        neg_score = sample['anger'] + sample['sorrow']
        if pos_score > neg_score:
            computed_sentiment = 'positive'
        elif neg_score > pos_score:
            computed_sentiment = 'negative'
        else:
            computed_sentiment = 'neutral'
        if computed_sentiment != sample['sentiment']:
            sample['sentiment'] = computed_sentiment

        # Append sample directly (no splitting)
        cleaned_data.append(sample)

    return cleaned_data

if __name__ == '__main__':
    # Load data
    input_file = 'merged_data.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Clean data
    cleaned_data = clean_data(data)

    # Save cleaned data
    output_file = 'cleaned_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"Data cleaning completed, saved to {output_file}")
