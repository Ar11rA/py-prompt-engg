import json

from openai import OpenAI

from vars import API_KEY

client = OpenAI(
    api_key=API_KEY
)

# string to json example
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You will be provided with a description of a mood, and your task is to generate the CSS code for a color that matches it. Write your output in json with a single key called \"hexCode\"."
        },
        {
            "role": "user",
            "content": "Blue sky at dusk."
        }
    ],
    temperature=0.7,
    max_tokens=64,
    top_p=1
)

res_txt = response.choices[0].message.content
res_json = json.loads(res_txt)
print(res_json)
print(res_json["hexCode"])

# direct json example by function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_color_by_mood",
            "description": "Get the color hex and text with the description of the mood",
            "parameters": {
                "type": "object",
                "properties": {
                    "hexCode": {
                        "type": "string",
                        "description": "The hex color code",
                    },
                    "colorText": {
                        "type": "string",
                        "description": "The color in normal text like red, blue, green etc",
                    }
                },
                "required": ["hexCode", "colorText"],
            },
        }
    }
]

completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """
                You will be provided with a description of a mood, and your task is to generate the CSS code for a color that matches it.
                """
        },
        {
            "role": "user",
            "content": "Blue sky at dusk."
        }
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.8,
    max_tokens=64,
    top_p=1
)

for call in completion.choices[0].message.tool_calls:
    response_args = json.loads(call.function.arguments)
    print(response_args)
