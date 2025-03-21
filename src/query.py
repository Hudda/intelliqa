from openai_client import generate_embeddings, bm25_encoder, openai_client
from pinecone_client import hybrid_index
from prompts.sustainability_prompt import SUSTAINABILITY_PROMPT


def get_answer(question, company_name):
    # Generate dense vector for the query
    query_dense_vector = generate_embeddings([question])[0].embedding

    # Generate sparse vector for the query
    query_sparse_vector = bm25_encoder.encode_queries(question)

    # Define the alpha parameter (balance between dense and sparse)
    alpha = 0.9  # Adjust as needed: 1.0 for purely dense, 0.0 for purely sparse

    # Perform the hybrid search
    results = hybrid_index.query(
        vector=[val * alpha for val in query_dense_vector],
        sparse_vector={
            'indices': query_sparse_vector['indices'],
            'values': [val * (1 - alpha) for val in query_sparse_vector['values']]
        },
        top_k=10,
        filter={'company_name': company_name},
        include_metadata=True
    )

    sustainability_data = ""
    for data in results["matches"]:
        sustainability_data += data["metadata"]["embedding"] + '\n'

    user_query = f"Sustainability Data:\n{sustainability_data}\nUserQuery:{question}"

    completion = openai_client.chat.completions.create(
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="gpt-4o",
        max_tokens=2048,
        messages=[
            {"role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SUSTAINABILITY_PROMPT,
                }
            ],
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )

    response = completion.choices[0].message.content.strip()

    return response