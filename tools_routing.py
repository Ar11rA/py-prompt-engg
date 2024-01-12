import datetime
from typing import Dict

import openai
import requests
from langchain.agents import tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vars import API_KEY

openai.api_key = API_KEY


@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"


print(search.args)


class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"


print(search.args)

print(search.run("sf"))


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


print(get_current_temperature.description)
print(get_current_temperature.args)
print(get_current_temperature({"latitude": 33, "longitude": 45}))
print(format_tool_to_openai_function(get_current_temperature))

# just tool to function
model = ChatOpenAI(temperature=0).bind(
    functions=[format_tool_to_openai_function(get_current_temperature)],
    function_call={"name": "get_current_temperature"}
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on world geography and locations."),
    ("user", "Tell me the latitude and longitude of: {place}")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

result = chain.invoke({"place": "New Delhi"})
print(result.tool_input)
print(get_current_temperature(result.tool_input))


# get final o/p
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)


chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route
result = chain.invoke({"place": "Bangalore"})
print(result)

# normal pydantic way
model = ChatOpenAI(temperature=0).bind(
    functions=[convert_pydantic_to_openai_function(OpenMeteoInput)],
    function_call={"name": "OpenMeteoInput"}
)

chain = prompt | model | JsonOutputFunctionsParser()

result = chain.invoke({"place": "Bangalore"})
print(result)
print(get_current_temperature(result))
