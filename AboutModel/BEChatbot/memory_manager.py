# memory_manager.py
import logging
from typing import Dict
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage # To help format history

import config # Import our config file

# In-memory store for session memory objects
# In a production/scaled environment, this might be replaced by Redis or similar
session_memory_store: Dict[str, ConversationBufferWindowMemory] = {}

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Gets or creates a ConversationBufferWindowMemory for the given session ID."""
    if session_id not in session_memory_store: #Check for if already a memory object for this session
        session_memory_store[session_id] = ConversationBufferWindowMemory(
            k=config.MEMORY_K, #Set the windows size based on config, using 6 to keep the context relevant and concise
            return_messages=True # Return BaseMessage objects
        )
        logging.info(f"Created new memory buffer for session: {session_id}") #Logs when a new memory buffer is created
    return session_memory_store[session_id]

def format_chat_history_for_prompt(messages: list[BaseMessage]) -> str:
    """Formats LangChain BaseMessage objects into a string suitable for an LLM prompt."""
    history_str = ""
    for msg in messages:
        # Langchain uses 'human' and 'ai' internally for message types
        role = "User" if msg.type == "human" else "AI"
        history_str += f"{role}: {msg.content}\n"
    return history_str.strip()

def update_memory(session_id: str, user_query: str, ai_response: str):
    """Updates the memory for a given session."""
    memory = get_session_memory(session_id)
    # Langchain memory expects inputs/outputs typically
    memory.save_context({"input": user_query}, {"output": ai_response})
    logging.debug(f"Updated memory for session: {session_id}")