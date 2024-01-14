from typing import Optional, List

import openai
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

openai.api_key = "API_KEY"


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


tagging_fn = convert_pydantic_to_openai_function(Tagging)
model = ChatOpenAI(temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model_with_functions = model.bind(
    functions=[tagging_fn],
    function_call={"name": "Tagging"}
)

tagging_chain = prompt | model_with_functions

print(tagging_chain.invoke({"input": "I love langchain"}))
print(tagging_chain.invoke({"input": "non mi piace questo cibo"}))

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

print(tagging_chain.invoke({"input": "non mi piace questo cibo"}))


class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")


class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")


extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})
print(extraction_model.invoke("Joe is 30, his mom is Martha"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))
