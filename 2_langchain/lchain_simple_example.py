import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI


prompt_tmpl = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt_tmpl | model | output_parser

res = chain.invoke({"topic": "dogs"})
print(res)
