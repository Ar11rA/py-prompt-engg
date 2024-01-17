import asyncio

import openai
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"topic": "bears"}))
print(chain.batch([{"topic": "bears"}, {"topic": "frogs"}]))
for t in chain.stream({"topic": "bears"}):
    print(t)


async def async_invoke():
    response = await chain.ainvoke({"topic": "bears"})
    print(response)


asyncio.run(async_invoke())
