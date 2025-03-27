import os
import concurrent.futures

from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

from constant import OPENAI_EMBEDDING_MODEL

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_embedding_client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=5,
    )

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=3,
    timeout=60,
    )

bm25_encoder = BM25Encoder.default()

def generate_embeddings(embedding_sources, model=OPENAI_EMBEDDING_MODEL):
    return openai_embedding_client.embeddings.create(input=embedding_sources, model=model).data


def get_embeddings_with_sparse(elements):
    embedding_sources = [element["metadata"]["embedding"] for element in elements]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_embeddings, embedding_sources),
            executor.submit(bm25_encoder.encode_documents, embedding_sources)
        ]
        results = [future.result() for future in futures]
    
    [embeddings, sparse_embeddings] = results

    filtered_elements = []
    for i in range(len(elements)):
        if len(sparse_embeddings[i]["values"]) > 0:
            elements[i]["values"] = embeddings[i].embedding
            elements[i]["sparse_values"] = sparse_embeddings[i]
            filtered_elements.append(elements[i])
    
    return filtered_elements

