from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_openai import ChatOpenAI

from models.location import Location


def get_location_summary(location: str):
    location_function = convert_pydantic_to_openai_function(Location)

    # get chat history from session
    # chat_repository.get_history(user_id, chat_id)

    # search vector db for query
    # chat_repository.get_relevant_docs(query)

    # add above data to system

    prompt = ChatPromptTemplate.from_template(
        """ 
        The place is {location} 
        """
    )

    # check if we can move this to config
    model = ChatOpenAI().bind(functions=[location_function])
    chain = prompt | model | JsonOutputFunctionsParser()

    # we can create agent here in case function call is required
    res = chain.invoke({
        "location": location
    })

    # save response to chat history
    return res

