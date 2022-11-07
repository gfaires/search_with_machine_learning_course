import pandas as pd
from pathlib import Path

# Read labeled products into a single column dataframe.  
products = pd.read_csv('~/workspace/corise/datasets/fasttext/labeled_products.txt', sep="##",engine="python",header=None)

# Split column on first white space
products = products[0].str.split(n=1, expand=True)
products = products.rename(columns={0:"label",1:"title"})
p = products.groupby('label')
filtered = p.filter(lambda x: x['title'].count() >= 500)

# Write pruned labeled products
output_path = Path('~/workspace/corise/datasets/fasttext/pruned_labeled_products.txt').expanduser()
with open(output_path, 'w') as file:
     for row in filtered.itertuples():
         file.write(f'{row.label} {row.title}\n')