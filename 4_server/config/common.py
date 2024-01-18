from langchain.prompts import ChatPromptTemplate

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system", """
                Please summarize the text into 2-3 sentences.
            """
    ),
    ("user", "{context}"),
])
