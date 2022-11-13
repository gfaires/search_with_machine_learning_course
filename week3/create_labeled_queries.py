import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")
general.add_argument("--categories", default=categories_file_name,  help="The full path to the filename containing the categories")
general.add_argument("--queries", default=queries_file_name,  help="The full path to the filename containing the queries training data")

args = parser.parse_args()
output_file_name = args.output
categories_file_name = args.categories
queries_file_name = args.queries

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize(query):
    normalized = re.sub('\W', ' ', query)
    normalized = re.sub('\s+', ' ', normalized)
    return stemmer.stem(normalized.lower())

print("Normalizing queries")
queries_df['query'] = queries_df['query'].apply(normalize)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

def categories_under_threshold(df, threshold=min_queries):
    return df.groupby("category").filter(lambda x: len(x) < threshold)['category'].unique()

print(f'Rolling up categories with threshold: {min_queries}')
cats_under_threshold = categories_under_threshold(queries_df)
while(len(cats_under_threshold) > 0):
    queries_df = pd.merge(queries_df, parents_df, on="category")
    queries_df["category"].mask(queries_df['category'].isin(cats_under_threshold),queries_df['parent'],inplace=True)
    cats_under_threshold = categories_under_threshold(queries_df)
    queries_df.drop('parent',axis=1,inplace=True)
    print(f'Number of categories left to roll-up: {len(cats_under_threshold)}')

category_count = queries_df["category"].nunique()
print(f'Number of unique categories: {category_count}')

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
