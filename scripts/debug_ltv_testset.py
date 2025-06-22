import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
import numpy as np

# Load processed customer sequences
df = pd.read_parquet('data/processed/customer_sequences.parquet')

# Use the same split as the data module
from utils import split_data

train, val, test = split_data(df, test_size=0.2, val_size=0.1, random_state=42)

print(f"Test set size: {len(test)}")
print("LTV summary statistics:")
print(test['ltv_90d'].describe())
print("\nSample LTV values:")
print(test['ltv_90d'].head(20).to_list())

print("\nAny negative LTVs?", (test['ltv_90d'] < 0).any())
print("Any all-zero LTVs?", (test['ltv_90d'] == 0).all())
print("Number of zero LTVs:", (test['ltv_90d'] == 0).sum()) 