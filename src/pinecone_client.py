import os
import nltk

from pinecone.grpc import PineconeGRPC
from constant import PINECONE_INDEX
from dotenv import load_dotenv

load_dotenv()

nltk.download('punkt_tab')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pinecone_client = PineconeGRPC(api_key=PINECONE_API_KEY)
hybrid_index = pinecone_client.Index(PINECONE_INDEX)
