import openai
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

openai.api_key = "API_KEY"

model = ChatOpenAI()
output_parser = StrOutputParser()

db = DocArrayInMemorySearch.from_texts(texts=[
    "Harry is a wizard",
    "Harry studies at Hogwarts",
    "Harry is smart, his best friends are Hermoine & Ron",
    "There are 4 houses in Hogwarts, Gryffindor, Slytherin, Ravenclaw, Hufflepuff"
], embedding=OpenAIEmbeddings())

print('****************')
print(db.similarity_search('Where does Harry study?'))
print(db.similarity_search_with_score('Where does Harry study?'))

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

inputs = RunnableMap({
    "context": lambda x: db.similarity_search(x["question"]),
    "question": lambda x: x["question"]
})

print('****************')
print(inputs.invoke({"question": "Where does Harry study?"}))

chain = inputs | prompt | model | output_parser

print('****************')
print(chain.invoke({"question": "Where does Harry study?"}))
