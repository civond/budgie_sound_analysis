import pandas as pd

df = pd.read_csv('meta_color.csv')


ls = []

train_df = df[df['fold'].isin([0,1,2,3])]
train_df['fold'] = pd.cut(train_df.index, bins=5, labels=False)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df  = train_df .sort_values(by='fold').reset_index(drop=True)

test_set = df[df['fold'] == 4].reset_index(drop=True)
test_set['fold'] = 5

ls.append(train_df)
ls.append(test_set)

merged_df = pd.concat(ls, ignore_index=True)

merged_df.to_csv("meta_color3.csv", index=False)
print(merged_df)
print(test_set)

print(train_df)