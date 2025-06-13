# database.py
# Setup neccessary database infracstructure
import sqlite3
import logging
import uuid
import config # Import our config file

def init_db():
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH) #Retrieve data from config.py
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL, -- 'user' or 'ai'
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"SQLite database initialized at {config.SQLITE_DB_PATH}")
    except Exception as e:
        logging.error(f"Failed to initialize SQLite DB: {e}", exc_info=True)

def add_message(session_id: str, sender: str, message: str):
    """Adds a message to the persistent chat history."""
    if not config.SQLITE_DB_PATH: 
        return
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        message_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO chat_history (id, session_id, sender, message) VALUES (?, ?, ?, ?)",
            (message_id, session_id, sender, message)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Cant adding message {session_id}: {e}", exc_info=True)

init_db()