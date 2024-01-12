from openai import OpenAI

from vars import API_KEY

client = OpenAI(
    organization='org-pYdGzOPN43a6LFq9DaFwD8Dp',
    api_key=API_KEY
)

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Hello!"}
    ]
)

print('\n')
print('***************')
print(completion.choices[0].message)

USER_INPUT = "I was really happy with the gift!"
CONTENT = "Classify the following text: {PROMPT}"
FINAL_PROMPT = CONTENT.format(PROMPT=USER_INPUT)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": FINAL_PROMPT
    }]
)

print('***************')
print(response.choices[0].message)
