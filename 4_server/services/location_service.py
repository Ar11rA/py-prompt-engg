from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI

from tools.location import get_location_coordinates


def get_location_summary(query: str):
    location_function = format_tool_to_openai_function(get_location_coordinates)

    # get chat history from session
    # chat_repository.get_history(user_id, chat_id)

    # search vector db for query
    # chat_repository.get_relevant_docs(query)

    # add above data to system

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", """
                You are a helpful assistant. 
                If your answer has multiple lines, please format it in bullet points. 
                Use html formatting for response in that case if it's not a tool call.
            """
        ),
        ("user", "{query}"),
    ])

    # check if we can move this to config
    model = ChatOpenAI(model_name="gpt-4-1106-preview")\
        .bind(functions=[location_function])
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()
    agent_executor = AgentExecutor(agent=chain, tools=[
        get_location_coordinates
    ], verbose=True)
    # we can create agent here in case function call is required
    res = agent_executor.invoke({
        "query": query
    })

    # save response to chat history
    return res
