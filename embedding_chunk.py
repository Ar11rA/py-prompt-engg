from openai import OpenAI

from vars import API_KEY

client = OpenAI(
    organization='org-pYdGzOPN43a6LFq9DaFwD8Dp',
    api_key=API_KEY
)


def get_embedding(text, model="text-embedding-ada-002"):
    res = client.embeddings.create(model=model, input=text)
    return res.data[0].embedding


def read_and_overlap_chunks(file_path, chunk_size=10, overlap=5):
    chunks = []
    with open(file_path, 'r') as file:
        content = file.read()
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = start + (chunk_size - overlap)

    return chunks


file_path = 'resources/info.txt'
chunk_size = 30
overlap = 3

chunks_list = read_and_overlap_chunks(file_path, chunk_size, overlap)
transformed_chunks = list(map(get_embedding, chunks_list))
print(transformed_chunks)
