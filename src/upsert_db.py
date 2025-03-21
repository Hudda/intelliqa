import concurrent.futures

from openai_client import get_embeddings_with_sparse
from utils import get_metadata
from pinecone_client import hybrid_index

def upsert_in_pinecone(chunks, company_name):
    chunks_with_metadata = [get_metadata(chunk, company_name) for chunk in chunks]

    with concurrent.futures.ThreadPoolExecutor(max_workers=35) as executor:
        futures = [executor.submit(get_embeddings_with_sparse, chunks_with_metadata[i:i+1000]) for i in range(0, len(chunks_with_metadata), 1000)]

        all_embeddings = [future.result() for future in concurrent.futures.as_completed(futures)]

    
    # Flatten all_embeddings
    flat_embeddings = [item for sublist in all_embeddings for item in sublist]


    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(hybrid_index.upsert, vectors=flat_embeddings[i:i+100]) for i in range(0, len(flat_embeddings), 100)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
