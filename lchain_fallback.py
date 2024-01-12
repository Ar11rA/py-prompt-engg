import json

from langchain.llms import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# using older model for error
simple_model = OpenAI(
    temperature=0,
    max_tokens=1000,
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads
challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"
# to show unstructured data
# print(simple_model.invoke(challenge))
# simple_chain.invoke(challenge) --> fails

model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads
print(chain.invoke(challenge))

final_chain = simple_chain.with_fallbacks([chain])
final_chain.invoke(challenge)
