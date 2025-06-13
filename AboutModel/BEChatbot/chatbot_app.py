# chatbot_app.py
# (Refined and Simplified Version)

import logging
import uuid
import io
import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import httpx
import chromadb
from fastapi.middleware.cors import CORSMiddleware

# Import custom modules
import config
import ai_models
import database
import memory_manager

#Logging Setup
logging.basicConfig(level=config.LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App & Startup Logic ---
app = FastAPI(title="Tomato AI Chatbot", version="1.0.0")

# CORS Middleware to allow frontend access
origins = ["http://localhost", "http://localhost:8080", "http://127.0.0.1", "http://127.0.0.1:8080", "null"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models and initialize database connections on application startup."""
    try:
        logger.info("Application Startup: Loading AI models...")
        ai_models.load_embedding_model()
        ai_models.load_image_model()

        logger.info(f"Application Startup: Connecting to ChromaDB at {config.CHROMA_DATA_PATH}")
        global chroma_client, knowledge_collection
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DATA_PATH)
        knowledge_collection = chroma_client.get_collection(name="tomato_chatbot_knowledge")
        logger.info(f"ChromaDB connected. Collection '{knowledge_collection.name}' has {knowledge_collection.count()} items.")

        logger.info("Application Startup: SQLite DB initialization checked.")
        # database.init_db() is called when its module is first imported.
    except Exception as e:
        logger.critical(f"CRITICAL STARTUP FAILURE: {e}", exc_info=True)
        # In a real production app, you might want to `raise SystemExit(e)` to stop startup.

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ImageAnalysisResponse(BaseModel):
    disease: str | None = None
    confidence: float | None = None
    error: str | None = None

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Tomato AI Chatbot Backend (AgriBot) is running!"}


@app.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """Receives an image and uses the trained Keras model to predict the disease."""
    logger.info(f"Received request for image analysis: {file.filename}")
    if not ai_models.image_model:
         logger.warning("Image analysis requested but model not loaded.")
         return ImageAnalysisResponse(error="Image analysis model is not available.")
    try:
        contents = await file.read()
        if not contents:
            return ImageAnalysisResponse(error="Received empty file.")

        img = Image.open(io.BytesIO(contents))
        disease, confidence = ai_models.predict_image_class(img)

        if disease is not None:
            return ImageAnalysisResponse(disease=disease, confidence=confidence)
        else:
            return ImageAnalysisResponse(error="Could not analyze image or prediction failed.")
    except UnidentifiedImageError:
        return ImageAnalysisResponse(error="Unsupported or invalid image file format.")
    except Exception as e:
        logger.error(f"Unexpected error during image analysis for '{file.filename}': {e}", exc_info=True)
        return ImageAnalysisResponse(error="An unexpected server error occurred.")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handles chat logic: targeted RAG search, prompt construction, and LLM call."""
    session_id = request.session_id or str(uuid.uuid4())
    full_query = request.query # The text from the frontend (may include image analysis prefix)
    logger.info(f"Chat request for session {session_id}: '{full_query[:200]}...'")

    database.add_message(session_id=session_id, sender='user', message=full_query)

    # --- Step 1: Refine RAG Search Query for Better Context ---
    rag_search_query = full_query
    match = re.search(r"Image analysis identified: ([\w\s.-]+)", full_query, re.IGNORECASE)
    if match:
        disease = match.group(1).strip()
        user_text_part = full_query.split("User query:")[-1].strip().lower()
        
        # Make the RAG search query more specific based on user's intent keywords
        intent_keywords = {"symptom": "symptoms", "treat": "treatment", "prevent": "prevention", "Cause_Explanation": "Cause_Explanation", "afternoon": "afternoon", 
                           "morning": "morning", "evening": "evening", "night": "night","Cause_Explanation_More": "Cause_Explanation_More","Cause_Explanation_More_1": "Cause_Explanation_More_1", 
                           "Saved_Plants": "Saved_Plants", "Saved_Plants_More": "Saved_Plants_More", "Symptoms Identification": "Symptoms Identification", "impact":"impact","impact_more":"impact_more",
                           "prevention":"prevention","prevention_more":"prevention_more","Prevention_future":"Prevention_future","Prevention_Guide":"Prevention_Guide","Saved_Plants_more":"Saved_Plants_more","Symptomps":"Symtomps",}
        detected_intent = "general information" # Default intent
        for key, value in intent_keywords.items():
            if key in user_text_part:
                detected_intent = value
                break
        
        rag_search_query = f"{detected_intent} of {disease}"
        logger.info(f"Refined RAG search query to: '{rag_search_query}'")

    # --- Step 2: Retrieve Context using Refined Query ---
    context_for_prompt = "No relevant information found in my knowledge base."
    try:
        query_embedding = ai_models.get_embedding(rag_search_query)
        if query_embedding:
            results = knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=config.N_RESULTS
            )
            if results and results.get('documents') and results['documents'][0]:
                context_for_prompt = "\n---\n".join(results['documents'][0])
                logger.info(f"Retrieved {len(results['documents'][0])} documents for RAG context.")
                logger.debug(f"CONTEXT for LLM:\n{context_for_prompt}")
        else:
            logger.warning(f"Could not generate embedding for RAG search: '{rag_search_query}'")
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
        # Continue with default context, LLM will handle it.

    # --- Step 3: Get History and Construct Final Prompt ---
    memory = memory_manager.get_session_memory(session_id)
    raw_history_messages = memory.chat_memory.messages
    history_lines = [f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in raw_history_messages]
    formatted_history = "\n".join(history_lines)

    system_instruction = """You are AgriBot, an expert AI assistant specializing in tomato plant diseases.
1. Your Name: If asked, state your name is AgriBot.
2. Core Task: Accurately answer user questions based *only* on the provided CONTEXT DOCUMENTS.
3. Formatting: For lists (symptoms, prevention), use a numbered list format. Be concise.
4. Handling Image Context: If the USER QUESTION includes "Image analysis identified:", acknowledge the disease, then answer the user's specific text query using the CONTEXT DOCUMENTS.
5. Handling No Context: If the CONTEXT DOCUMENTS do not contain the answer, clearly state "I do not have specific information about that in my knowledge base."
6. Off-Topic: If the question is not about tomato diseases, politely decline by stating you only specialize in tomato plant health.
Do not invent information."""

    prompt = f"""{system_instruction}

CONVERSATION HISTORY:
{formatted_history if formatted_history else "This is the start of the conversation."}

CONTEXT DOCUMENTS:
{context_for_prompt}

USER QUESTION:
{full_query}

ASSISTANT ANSWER:"""
    logger.debug(f"SENDING PROMPT TO OLLAMA (Session: {session_id}):\n{prompt}")

    # --- 4. Call LLM ---
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": config.OLLAMA_MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": { "temperature": 0.2 }
            }
            response = await client.post(config.OLLAMA_API_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()
            ai_response_text = response_data.get("response", "Sorry, I received an empty response from the AI.").strip()
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM service returned error status {e.response.status_code}: {e.response.text}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Could not connect to LLM service: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Could not connect to LLM service (AgriBot's brain).")
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while thinking.")
    
    # --- 5. Update Memory & DB ---
    if "sorry" not in ai_response_text.lower() and "error" not in ai_response_text.lower():
        database.add_message(session_id=session_id, sender='ai', message=ai_response_text)
        memory_manager.update_memory(session_id, full_query, ai_response_text)

    # --- 6. Return Response ---
    return ChatResponse(response=ai_response_text, session_id=session_id)