# using openai directly
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "system",
            "content": """ 
            Your job is to analyse charts and generate meaningful insights.
            Make sure that any trends are pointed out.
            If the image is not a graph, reply with NO_GRAPH_FOUND
            If you cannot understand the graph, just reply with DATA_UNREADABLE.
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "There are 2 charts below. What can we make of them?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d4/Graph_of_the_number_of_mass_incidents_in_China_from_1993_to_2006.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/Nairn_Religion_Pie_Chart.png",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/8/85/Vera_Pollo_belarusian_soviet_actress.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=800,
)

print(response.choices[0].message)
print('*************************')

# using langchain basic
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=800)
res = chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is this image showing"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/Nairn_Religion_Pie_Chart.png",
                        "detail": "auto",
                    },
                },
            ]
        )
    ]
)

print(res)
print('*************************')

# using langchain with chain syntax
# gpt 4 vision does not support function calling as of now
from langchain_openai import ChatOpenAI


# create the chain
model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=800)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content=[{
            "type": "text",
            "text": """"
            Your job is to analyse charts and generate meaningful insights.
            Make sure that any trends are pointed out.
            If the image is not a graph, reply with NO_GRAPH_FOUND
            If you cannot understand the graph, just reply with DATA_UNREADABLE.
            """
        }],
    ),
    HumanMessage(
        content=[
            {"type": "text", "text": "Can you generate insights for the image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/7/7a/Nairn_Religion_Pie_Chart.png",
                    "detail": "auto",
                },
            },
        ]
    )
])
chain = prompt | model


print(chain.invoke({}))
