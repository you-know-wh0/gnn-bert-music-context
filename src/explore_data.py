import pandas as pd
import ast

df = pd.read_csv('data/musiccaps/musiccaps-public.csv')
print("Total clips:", df.shape[0])
print("\nColumns:", list(df.columns))

df['aspect_list'] = df['aspect_list'].apply(ast.literal_eval)

print("\n--- Example row ---")
print("Caption:", df['caption'].iloc[0])
print("Tags:", df['aspect_list'].iloc[0])