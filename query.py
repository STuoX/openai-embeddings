import os
import time
import pandas as pd
import numpy as np
from collections import namedtuple
from utils import openai_auth
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity


DOCUMENTS_EMBEDDINGS_PATH = "data"  # a folder with all the documents embeddings. within this folder, one csv file include multiple documents embedding of the same run
COLUMN_EMBEDDINGS = "embedding"  # the embedding column name in the documents embedding file.

# embedding model parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
ENCODING_MODEL = "cl100k_base"  # this is the encoding for text-embedding-ada-002
MAX_TOKENS = 8191  # the maximum for text-embedding-ada-oo2 is 8191

def embed_query(query:str):
    # embed query
    embedding = get_embedding(query, engine=EMBEDDING_MODEL)
    return embedding

def pick_a_file_from_folder(folder_path:str):
    # pick a file from a folder
    files = os.listdir(folder_path)
    print(f'Found {len(files)} files in {folder_path}')
    for i, file in enumerate(files):
        print(f'{i+1}. {file}')
    file_index = int(input('Please enter the number of the document embedding file you want to use: '))
    file_path = os.path.join(folder_path, files[file_index-1])
    return file_path

def read_documents_embeddings(file_path:str):
    # read documents embeddings
    df = pd.read_csv(file_path)
    df[COLUMN_EMBEDDINGS] = df[COLUMN_EMBEDDINGS].apply(eval).apply(np.array) # convert string to np array
    print(f'Read {len(df)} documents embeddings from {file_path}')
    return df

def get_query_from_user():
    # get query from user
    query = input('Please enter a query: ')
    return query

def compute_similarity(df, query_embedding):
    # compute similarity
    df['similarity'] = df[COLUMN_EMBEDDINGS].apply(lambda x: cosine_similarity(x, query_embedding))
    return df

if __name__ == '__main__':
    file_path = pick_a_file_from_folder(folder_path=DOCUMENTS_EMBEDDINGS_PATH)
    df = read_documents_embeddings(file_path=file_path)
    openai_auth()
    query=""
    while query != "exit":
        print('***********************************')
        print('Enter "exit" to exit the script')
        query = get_query_from_user()
        if query == "exit":
            break
        tic = time.time()
        query_embedding = embed_query(query=query)
        toc = time.time()
        print(f'Embedding query took {round(toc-tic)*1000}ms')
        print('Top matches ordered by cosine similarity of vector embeddings:')
        df = compute_similarity(df, query_embedding)
        print(df.sort_values(by='similarity', ascending=False)[['similarity', 'description']].head(10))
