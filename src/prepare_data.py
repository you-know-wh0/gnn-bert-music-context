import pandas as pd
import numpy as np
import ast
import json
from collections import Counter
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/musiccaps/musiccaps-public.csv')
df['aspect_list'] = df['aspect_list'].apply(ast.literal_eval)

tag_counts = Counter()
for tags in df['aspect_list']:
    tag_counts.update(tags)
top_k_tags = [t for t, _ in tag_counts.most_common(50)]

with open('data/musiccaps/label_vocab.json', 'w') as f:
    json.dump(top_k_tags, f, indent=2)
print("Saved 50 label vocabulary to data/musiccaps/label_vocab.json")
print("Top 10 tags:", top_k_tags[:10])

def multi_hot(tags, vocab):
    tag_set = set(tags)
    return [1.0 if t in tag_set else 0.0 for t in vocab]

df['label_vector'] = df['aspect_list'].apply(lambda t: multi_hot(t, top_k_tags))

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df[['caption', 'label_vector']].to_json('data/musiccaps/train.jsonl', orient='records', lines=True)
val_df[['caption', 'label_vector']].to_json('data/musiccaps/val.jsonl', orient='records', lines=True)
test_df[['caption', 'label_vector']].to_json('data/musiccaps/test.jsonl', orient='records', lines=True)

print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print("Saved splits to data/musiccaps/{train,val,test}.jsonl")