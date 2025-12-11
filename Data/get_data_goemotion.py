from datasets import load_dataset
import pandas as pd

ds = load_dataset("google-research-datasets/go_emotions", "simplified")
# show all ds keys
print(ds.keys())

# for train csv
train_df = pd.DataFrame(ds['train'])
train_df.to_csv("go_emotions_simplified_train.csv", index=False)

# for test csv
test_df = pd.DataFrame(ds['test'])
test_df.to_csv("go_emotions_simplified_test.csv", index=False)

# for validation csv
val_df = pd.DataFrame(ds['validation']) 
val_df.to_csv("go_emotions_simplified_validation.csv", index=False)