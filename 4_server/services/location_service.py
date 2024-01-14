from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI

from models.location import Location


def get_location_summary(location: str):
    color_function = convert_pydantic_to_openai_function(Location)

    prompt = ChatPromptTemplate.from_template(
        """ 
        The place is {location} 
        """
    )
    model = ChatOpenAI().bind(functions=[color_function])
    chain = prompt | model | JsonOutputFunctionsParser()
    return chain.invoke({
        "location": location
    })

