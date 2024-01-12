import datetime
from typing import Dict

import openai
import requests
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor

from vars import API_KEY

openai.api_key = API_KEY


# Define the input schema
class OpenMeteoInput(BaseModel):
    """Input for location"""
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput, return_direct=True)
def get_current_temperature(latitude: float, longitude: float) -> Dict:
    """Fetch current temperature for given coordinates."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'


# just tool to function
model = ChatOpenAI(temperature=0).bind(
    functions=[format_tool_to_openai_function(get_current_temperature)],
)

print('*********', model.invoke("What is temperature of Delhi?"))
print('**********', model.invoke("Hi!"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history"),
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

print(chain.invoke({"input": "What is temperature of Delhi?", "agent_scratchpad": [], "chat_history": []}))
print(chain.invoke({"input": "Hi!", "agent_scratchpad": [], "chat_history": []}))

tools = [get_current_temperature]
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)
print(agent_executor.invoke({"input": "What is temperature of Bangalore?"}))
print(agent_executor.invoke({"input": "What is temperature of Moscow?"}))
print(agent_executor.invoke({"input": "What is latitude and longitude of Moscow?"}))
print(agent_executor.invoke({"input": "Hi, I am Aritra"}))
print(agent_executor.invoke({"input": "Hi, Who am I?"}))
