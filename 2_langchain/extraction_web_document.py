from typing import Optional, List

import openai
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

openai.api_key = "API_KEY"

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]


class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: List[str] = Field(description="Provide keywords related to the content.")


class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]


overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
model = ChatOpenAI(temperature=0)
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name": "Overview"}
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()
print(tagging_chain.invoke({"input": page_content}))

paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function,
    function_call={"name": "Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
print(extraction_chain.invoke({"input": page_content}))

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
print(extraction_chain.invoke({"input": page_content}))

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = text_splitter.split_text(doc.page_content)
print(len(splits))


def split_ctx(x):
    arr = []
    for docx in text_splitter.split_text(x):
        arr.append({"input": docx})
    return arr


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


prep = RunnableLambda(
    lambda x: [{"input": docx} for docx in text_splitter.split_text(x)]
)
inputs = prep.invoke(doc.page_content)
print(len(inputs))

chain = prep | extraction_chain.map() | flatten
print(chain.invoke(doc.page_content))
