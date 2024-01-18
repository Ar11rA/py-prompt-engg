from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.common import summary_prompt
from repository.chat_history_repository import load_chat_history
from repository.knowledge_repository import query_documents
from tools.location import get_location_coordinates


# keeping this here in case specific summary is needed
def _summarize_chat_history(chat_id: str) -> str:
    messages = load_chat_history(chat_id)
    chat_history = messages_from_dict(messages)
    model = ChatOpenAI(model_name="gpt-4-1106-preview")
    summary_chain = summary_prompt | model | StrOutputParser()
    chat_summary = summary_chain.invoke({"context": chat_history})
    return chat_summary


def get_location_summary(query: str) -> str:
    location_function = format_tool_to_openai_function(get_location_coordinates)

    # get chat history from session, hardcoding session for easy understanding
    # get chat id from user info using middleware
    chat_summary = _summarize_chat_history("1")

    # search vector db for query, hardocoding collection name for easy understanding
    # get collection name based on use case
    reference_docs = query_documents("learning", query)

    # add above data to system context
    # have use case specific prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system", """
                You are a helpful assistant. 
                If your answer has multiple lines, please format it in bullet points. 
                Use html formatting for response in that case if it's not a tool call.
                
                Use this for reference:
                {reference_docs}
                
                Here is the previous chat history:
                {chat_history}
            """
        ),
        ("user", "{query}"),
    ])

    # check if we can move this to config
    model = ChatOpenAI(model_name="gpt-4-1106-preview") \
        .bind(functions=[location_function])
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()
    agent_executor = AgentExecutor(agent=chain, tools=[
        get_location_coordinates
    ], verbose=True)
    # we can create agent here in case function call is required
    res = agent_executor.invoke({
        "query": query,
        "reference_docs": reference_docs,
        "chat_history": chat_summary
    })

    # save response to chat history
    return res['output']
