from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# loader = TextLoader('../resources/info.txt', encoding='utf-8')
# documents = loader.load()
# print(len(documents))
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
# texts = text_splitter.split_documents(documents)
#
# print(len(texts))
#
embeddings = OpenAIEmbeddings()
# doc_vectors = embeddings.embed_documents([t.page_content for t in texts])
#
# print(len(doc_vectors))
# print(len(doc_vectors[0]))

pg_connection_string = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres"
collection_name = "learning"
#
# # we are filling entries in the database here
# db = PGVector.from_documents(
#     embedding=embeddings,
#     documents=texts,
#     collection_name=collection_name,
#     connection_string=pg_connection_string
# )

# if you want to just connect to the database for querying etc
db_existing = PGVector(
    embedding_function=embeddings,
    collection_name=collection_name,
    connection_string=pg_connection_string
)

content = db_existing.similarity_search("What are fields in computer science?", k=10)
print(content)
print('***********************')

content = db_existing.similarity_search("What are fields in computer science?", k=10, filter={
    "source": "resources/new.txt"
})
print(content)

# content = db.similarity_search("What are fields in computer science?", k=10)
# model = ChatOpenAI(temperature=0)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#         You are a professor in computer science.
#         {context}
#     """),
#     ("user", "{input}"),
# ])
#
# print('**************************************')
# print(model.invoke(
#     [
#         SystemMessage(
#             content=[
#                 {"type": "text", "text": "You are a professor in computer science"
#                                          + ','.join([c.page_content for c in content])}
#             ]
#         ),
#         HumanMessage(
#             content=[
#                 {"type": "text", "text": """
#                 What are fields in computer science?
#                 Can you mention them in a list?
#                 Do not give any other text.
#                 """},
#             ]
#         )
#     ]
# ))
#
# print('**************************************')
# message = {
#     "context": ','.join([c.page_content for c in content]),
#     "input": """
#         What are fields in computer science?
#         Can you mention them in a list?
#         Do not give any other text.
#     """
# }
# print(prompt.invoke(message))
#
# chain = prompt | model
#
# print('**************************************')
# print(chain.invoke(message))
