from langchain.memory import PostgresChatMessageHistory
from langchain_core.messages import messages_to_dict

from config.env import PG_CONNECTION_STRING
from dto.interaction import Interaction


# session id is chat id for the below functions
def load_chat_history(session_id: str):
    db = PostgresChatMessageHistory(session_id, PG_CONNECTION_STRING)
    return messages_to_dict(db.messages)


def save_qa_to_history(session_id: str, interaction: Interaction):
    db = PostgresChatMessageHistory(session_id, PG_CONNECTION_STRING)
    db.add_user_message(interaction.question)
    db.add_ai_message(interaction.answer)
