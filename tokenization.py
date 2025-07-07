import bz2
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

file_path = '/home/admin1/Project/Amazon Review/Data/train.ft.txt.bz2'
file_path_1 = '/home/admin1/Project/Amazon Review/Data/test.ft.txt.bz2'


texts = []
labels = []

with bz2.open(file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 1)  # split label and text
        if len(parts) == 2:
            label, text = parts
            labels.append(label.replace('__label__', ''))  # remove FastText prefix
            texts.append(text)

texts_1 = []
labels_1 = []

with bz2.open(file_path_1, 'rt', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 1)  # split label and text
        if len(parts) == 2:
            label_1, text_1 = parts
            labels_1.append(label_1.replace('__label__', ''))  # remove FastText prefix
            texts_1.append(text_1)

df_train = pd.DataFrame({'label': labels, 'text': texts})
print(df_train.head())
df_test = pd.DataFrame({'label': labels_1, 'text': texts_1})
print(df_test.head())

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Convert pandas to Dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize with batching and multiprocessing
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Optionally remove unnecessary columns
tokenized_train = tokenized_train.remove_columns(['text'])
tokenized_test = tokenized_test.remove_columns(['text'])

# Save
tokenized_train.save_to_disk('tokenized_train_1')
tokenized_test.save_to_disk('tokenized_test_1')