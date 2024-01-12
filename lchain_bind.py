import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vars import API_KEY

openai.api_key = API_KEY

functions = [
    {
        "name": "weather_search",
        "description": "Search for weather given an airport code",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "The airport code to get the weather for"
                },
            },
            "required": ["airport_code"]
        }
    }
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)

runnable = prompt | model

print(runnable.invoke({"input": "what is the weather in sf"}))
print(runnable.invoke({"input": "what is the weather in new delhi"}))
