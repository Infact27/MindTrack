from datasets import load_dataset

ds = load_dataset("andreagasparini/dreaddit")

# save train and test to csv
train_df = ds['train'].to_pandas()
train_df.to_csv("dreaddit_train.csv", index=False)
test_df = ds['test'].to_pandas()
test_df.to_csv("dreaddit_test.csv", index=False)