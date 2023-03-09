# openai-embeddings

This is a demonstration of using OpenAI API to run neural search based on vector embeddings.
* A sample data of product descriptions was downloaded from: https://www.kaggle.com/datasets/cclark/product-item-data . Data in a similar format can be used as well.
* First, documents (products descriptions in this case) are embedded into a vector space (`generate_documents_embeddings.py`)
* Next, one can run queries on the embeddings (`query.py`)

The model used is OpenAI's `text-embedding-ada-002`, which was released on December 15, 2022 ([Release Notes](https://openai.com/blog/new-and-improved-embedding-model)).

## Setup
* OpenAI API Key is required. Create a `.env` file (use `.env.example` as a template), and enter your API Key there.