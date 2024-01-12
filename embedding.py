import numpy as np
import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from vars import API_KEY

client = OpenAI(
    organization='org-pYdGzOPN43a6LFq9DaFwD8Dp',
    api_key=API_KEY
)


def get_embedding(text, model="text-embedding-ada-002"):
    res = client.embeddings.create(model=model, input=text)
    return res.data[0].embedding


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


source_text = """
    what is your age?
"""
print(num_tokens_from_string(source_text))
source_embedding = get_embedding(source_text)

input_text_1 = "How old are you?"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_1))]))
input_text_2 = "Hey, what's up?"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_2))]))
input_text_3 = "Boss up"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_3))]))
input_text_4 = "12"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_4))]))
input_text_5 = "Horrible player"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_5))]))
input_text_6 = "Kuch bhi irrelevant unreadable bakwas"
print(cosine_similarity([np.array(source_embedding)], [np.array(get_embedding(input_text_6))]))
