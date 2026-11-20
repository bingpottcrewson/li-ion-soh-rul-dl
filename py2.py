import pandas as pd

metadata = pd.read_csv('metadata.csv')
print(metadata.columns.tolist())
