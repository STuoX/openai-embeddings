import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding
from utils import openai_auth

# input data parameters
DOCUMENT_PATH = "data/cclark-product-descriptions.csv"
COLUMN_TO_EMBED = "description"

# output data parameters
OUTPUT_PATH = "data/cclark-documents-embeddings.csv"

#embedding model parameters
EMBEDDING_MODEL = "text-embedding-ada-002"
ENCODING_MODEL = "cl100k_base"  # this is the encoding for text-embedding-ada-002
MAX_TOKENS = 8191  # the maximum for text-embedding-ada-oo2 is 8191

def print_parameters():
    print(f'Using embedding model: {EMBEDDING_MODEL}')
    print(f'Using encoding: {ENCODING_MODEL}')
    print(f'Maximum number of tokens: {MAX_TOKENS}')
    print('***********************************')
    return

def read_documents(file_path:str):
    df = pd.read_csv(file_path)
    print(f'Read {len(df)} documents from {file_path}')
    return df

def generate_encodings(df:pd.DataFrame, column_to_encode:str):
    # tokenize: generate encodings for 'cloumn_to_encode'
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    df["encoding"] = df[column_to_encode].apply(lambda x: encoding.encode(x))
    print(f'Encoded the {column_to_encode} column into an encoding column')
    return df

def omit_long_encodings(df:pd.DataFrame, max_tokens:int):
    # omit encodings that are too long to embed
    df["n_tokens"] = df.encoding.apply(lambda x: len(x))
    n_long_encodings = len(df[df.n_tokens > max_tokens])
    df = df[df.n_tokens <= max_tokens]
    print(f'Omitted {n_long_encodings} encodings that were too long to embed')
    return df

def embed_encodings(df:pd.DataFrame, column_to_embed:str, embedding_model:str):
    # embed: generate embeddings for 'column_to_embed', by calling the OpenAI API
    df["embedding"] = df[column_to_embed].apply(lambda x: get_embedding(x, engine=embedding_model))
    print(f'Embedded encodings into an embedding column, by calling the OpenAI API')
    return df

def save_embeddings(df:pd.DataFrame, output_path:str):
    # save embeddings to output_path
    df.to_csv(output_path, index=False)
    print(f'Saved embeddings to {output_path}')
    return

if __name__ == '__main__':
    #pd.options.display.max_columns = None
    print_parameters()
    df = read_documents(file_path=DOCUMENT_PATH)
    df = generate_encodings(df=df, column_to_encode=COLUMN_TO_EMBED)  # optional, needed only for omitting long embeddings
    df = omit_long_encodings(df=df, max_tokens=MAX_TOKENS)  # optional, needed only for omitting long embeddings
    openai_auth()
    df = embed_encodings(df=df, column_to_embed=COLUMN_TO_EMBED, embedding_model=EMBEDDING_MODEL)
    save_embeddings(df=df, output_path=OUTPUT_PATH)
    print(df.head())