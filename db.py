import os
from typing import Final
from llama_cpp import Llama
from dotenv import load_dotenv
from pymilvus import MilvusClient


load_dotenv()

DB_EMBEDDING_PATH: Final[str | None] = os.getenv('DB_EMBEDDING_PATH')
DB_COLLECTION_NAME: Final[str | None] = os.getenv('DB_COLLECTION_NAME')

assert DB_EMBEDDING_PATH is not None
assert DB_COLLECTION_NAME is not None


class VectorDb:
    '''Milvus client for RAG.'''

    milvus_client = MilvusClient()
    embedding: Final[Llama] = Llama(
        model_path=DB_EMBEDDING_PATH,
        embedding=True,
        verbose=False
    )

    def get_embedding(self, query) -> str:
        try:
            return str(self.embedding.create_embedding(query)['data'][0]['embedding'])

        except Exception as e:
            return 'Sorry, embedding failed.'

    def query_milvus(self, embedding):
        result_count = 3

        result = self.milvus_client.search(
            collection_name=DB_COLLECTION_NAME,
            data=[embedding],
            limit=result_count,
            output_fields=["path", "text"])

        list_of_knowledge_base = list(map(lambda match: match['entity']['text'], result[0]))
        list_of_sources = list(map(lambda match: match['entity']['path'], result[0]))

        return {
            'list_of_knowledge_base': list_of_knowledge_base,
            'list_of_sources': list_of_sources
        }
