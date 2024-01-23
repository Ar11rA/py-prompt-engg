# Generative AI

## Overview

To build applications around GenAI, these are the 4 main building blocks:

1. Prompt Engineering
2. LLMs with Langchain - Chat completions, vision etc
3. Vector Databases
4. Context Management

## Pre requisites

To understand this repo, you should know the following:

- Intermediate knowledge of Python
- Basics of Data Science
- API understanding and development

## Tools

- Python 3.11, 3.12
- Postgres 12+

## Installation

```shell
# Install Dependencies
pip install poetry
poetry install

# Run a particular file
python 1_basics/basic.py 

# Run a full blown web server
cd 4_server
python app.py

# TODO - Add docker compose for 4_server
```

## Folder structure explanation

```shell
.
├── 1_basics - Get started with invoking openAI LLMs and understanding their API offerings.
├── 2_langchain - Understand this library to work with a wide range of LLMs and community packages. 
├── 3_vector_store - Work with vector databases and understand what embeddings are.
├── 4_server - Collate the learnings from above 3 into a structured framework and expose it via an API interface.
├── 5_llm_foundations - Understand in detail how LLMs work.
└── resources - Common files used for reading/parsing etc
```


## Deployment

For deployment, cloud resources needed

AWS:

1. Sagemaker - For hosting your own LLMs
2. Amazon Opensearch - Vector Database + Full text search
3. EKS/ECS - For hosting web apps 

Azure:

1. ML Studio - For hosting your own LLMs
2. Azure AI Search - Vector Database + Full text search
3. AKS/ACI - For hosting web apps 

On prem

1. ArgoWF/Temporal/K8s - For hosting your own LLMs
2. ChromaDB/Pinecone/Elasticsearch/PGVector - Vector Database + Full text search
3. K8s - For hosting web apps 

## Resources

- https://www.deeplearning.ai/ - Andrew Ng
- https://www.promptingguide.ai/
- https://platform.openai.com/usage