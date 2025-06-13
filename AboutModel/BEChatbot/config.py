# config.py
import os

#Define paths use in project
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Base directory of the project 
CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_db_store") # Path to store ChromaDB data
SQLITE_DB_PATH = os.path.join(BASE_DIR, "chat_history.db") # Path to store SQLite databas
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "VGG16_TomatoDiseases.keras") #Path to .keras model for predict images

#Model Names
#Specifies the exact name of the Sentence Transformer model to use for creating text embeddings.
#Purpose: The ai_models.py module will use this name to load the correct model 
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' #Embedding model
OLLAMA_MODEL_NAME = "phi" # Ensure this model is available in Ollama

#Define the class names for the image classification model, must be in the same order as the training labels.
IMAGE_CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Yellow Leaf Curl"
]

#API Endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama endpoint

#RAG & Memory Settings
N_RESULTS = 3 # Number of documents to retrieve from ChromaDB this determines how many similar documents to retrieve for context injection (i.e., in Retrieval-Augmented Generation).
MEMORY_K = 6 # Number of conversation turns (user + AI = 1 turn) to keep in windowed memory

#Image Model Settings 
IMAGE_TARGET_SIZE = (224, 224)

#Other Settings
LOGGING_LEVEL = "DEBUG" #"DEBUG", "INFO", "WARNING"
API_HOST = "0.0.0.0"
API_PORT = 8000