from langchain.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from config.env import PG_CONNECTION_STRING


def query_documents(collection_name: str, query: str, top_k: int = 10):
    embeddings = OpenAIEmbeddings()
    db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=PG_CONNECTION_STRING
    )
    content = db.similarity_search_with_score(query, top_k)
    return content
