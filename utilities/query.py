# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import sys
from pathlib import Path
import fasttext

import re
from sentence_transformers import SentenceTransformer
sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')
# model_path = Path('~/workspace/corise/datasets/fasttext/model_classification.bin').expanduser()
# print(f'Loading classification model from: {model_path}')
# model = fasttext.load_model(str(model_path))
similarity_threshold = 0.5

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, use_synonyms=False):
    name_field = "name.synonyms" if use_synonyms else "name"
    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    name_field: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": [f"{name_field}^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj

def normalize_query(query):
    normalized = re.sub('\W', ' ', query)
    normalized = re.sub('\s+', ' ', normalized)
    return stemmer.stem(normalized.lower())

def classify_query(query, threshold=similarity_threshold):
    print(f'Classifying query:{query}')
    normalized_query = normalize_query(query)
    print(f'Normalized query: {normalized_query}')
    
    k = 5
    label_prefix = "__label__"
    categories, probabilities = model.predict(query, k=k)
    categories = [category.removeprefix(label_prefix) for category in categories]
    print("Query classification results:")
    for category, probability in zip(categories, probabilities):
        print (category, probability)
    min_threshold = 0.5
    score = 0
    classified_categories = []
    for category, probability in zip(categories, probabilities):
        classified_categories.append(category)
        score += probability
        if (score >= min_threshold):            
            break
    else:
        print("Didn't get to threshold for classified categories")
    print(f'Classified with categories: {classified_categories} and combined score {score}')
    return classified_categories

def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", use_synonyms=False, use_vectors=False):
    if use_vectors:
        query_obj = create_vector_query(user_query=user_query)
    else:
        #### W3: classify the query
        classified_query_categories = classify_query(user_query)
        #### W3: create filters and boosts
        filters = []
        if (classified_query_categories is not None):
            filters.append(
                {
                    "terms": {
                        "categoryPathIds.keyword": classified_query_categories
                    }
                }
            ) 
        # Note: you may also want to modify the `create_query` method above
        query_obj = create_query(user_query, click_prior_query=None, filters=filters, sort=sort, sortDir=sortDir, source=["name", "shortDescription"], use_synonyms=use_synonyms)
    logging.info(query_obj)
    response = client.search(query_obj, index=index)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        print(json.dumps(response, indent=2))

def create_vector_query(user_query, num_results=10):
    vector = sentence_transformer.encode([user_query])[0]
    return {
        "size": num_results,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": num_results
                }
            }
        }
    }

if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument('--synonyms',
                         help='Use synonyms on name field', action='store_true')
    general.add_argument('--vector',
                         help='Use vector query', action='store_true')
    general.add_argument('--query',
                         help='User query')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    use_synonyms = args.synonyms
    use_vectors = args.vector
    index_name = args.index

    query = args.query.rstrip()

    search(client=opensearch, user_query=query, index=index_name, use_synonyms=use_synonyms, use_vectors=use_vectors)

    