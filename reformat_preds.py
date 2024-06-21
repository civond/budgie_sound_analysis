import pandas as pd
import numpy as np

pth = "csv/bl122_unseen_preds.csv"
txt = "csv/bl122_unseen.txt"
df = pd.read_csv(pth)

columns_to_keep = ['onset', 'offset', 'preds']
df = df.drop(df.columns.difference(columns_to_keep), axis=1)
df['onset'] = np.round(df['onset'],6)
df['offset'] = np.round(df['offset'],6)
df = df.sort_values(by='onset')
df.to_csv(txt, sep='\t',header=False, index=False)

print(df)