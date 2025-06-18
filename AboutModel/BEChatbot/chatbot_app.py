# chatbot_app.py
import logging #logging DEBUG INFO WARNING
import uuid #uuid for session ID generation
import io
import re
from fastapi import FastAPI, HTTPException, UploadFile, File #Using FastAPI to create chatbot
from pydantic import BaseModel 
from PIL import Image, UnidentifiedImageError #Use for image processing
import httpx
import chromadb
from fastapi.middleware.cors import CORSMiddleware #allow chatbot to run on different port

# Import custom modules
import config
import ai_models
import database
import memory_manager

#Logging Setup
logging.basicConfig(level=config.LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#FastAPI App & Startup Logic
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

#Pydantic Models
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

#API Endpoints
@app.get("/")
async def root():
    return {"message": "Tomato AI Chatbot Backend (AgriBot) is running!"}


@app.post("/analyze_image", response_model=ImageAnalysisResponse) #This function retrieves upload images from user to predict the tomato diseases using Keras model
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """Receives an image and uses the trained Keras model to predict the disease."""
    logger.info(f"Received request for image analysis: {file.filename}")
    if not ai_models.image_model: #Check the image_model is loaded
         logger.warning("Image analysis requested but model not loaded.")
         return ImageAnalysisResponse(error="Can't use the model.")
    try:
        contents = await file.read()
        if not contents:
            return ImageAnalysisResponse(error="Received empty file.")

        img = Image.open(io.BytesIO(contents))
        disease, confidence = ai_models.predict_image_class(img) #Call model to make the prediction of diseases 

        if disease is not None:
            return ImageAnalysisResponse(disease=disease, confidence=confidence)
        else:
            return ImageAnalysisResponse(error="Could not analyze image or prediction failed.")
    except UnidentifiedImageError:
        return ImageAnalysisResponse(error="Unsupported or invalid image file format.")
    except Exception as e:
        logger.error(f"Unexpected error during image analysis for '{file.filename}': {e}", exc_info=True)
        return ImageAnalysisResponse(error="An unexpected server error occurred.")


#Heart of our chatbot. Handle user queries, perform RAG, call the LLM and return the AI's response
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handles chat logic: targeted RAG search, prompt construction, and LLM call."""
    session_id = request.session_id or str(uuid.uuid4())
    full_query = request.query # The text from the frontend
    logger.info(f"Chat request for session {session_id}: '{full_query[:200]}...'")

    database.add_message(session_id=session_id, sender='user', message=full_query) #add user query to the database

    #Refine RAG Search Query for Better Context
    rag_search_query = full_query # Default
    user_text_part_for_intent = full_query # Use full_query for intent detection if no image prefix
    
    disease_context_present = False
    match = re.search(r"Image analysis identified: ([\w\s.-]+)", full_query, re.IGNORECASE)
    if match:
        disease = match.group(1).strip()
        user_text_part_for_intent = full_query.split("User query:")[-1].strip().lower()
        disease_context_present = True
        # Default RAG query if disease is present but no specific intent below matches
        rag_search_query = f"general information of {disease}"


    # Make the RAG search query more specific based on user's intent keywords
    intent_keywords = {
        # Greetings & Farewells mapped to Coms.csv intents
        "good morning": "Morning",       # Maps to "Morning" intent
        "good afternoon": "Afternoon",   # Maps to "Afternoon" intent
        "good evening": "Evening",     # Maps to "Evening" intent
        "good night": "Night",         # Maps to "Night" intent
        "hello": "Greetings",        # General greetings
        "hi": "Greetings",
        "hey": "Greetings",
        "goodbye": "Goodbye",
        "bye": "Goodbye",
        "see you": "Goodbye",
        "adios": "Goodbye",
        "talk to you later": "Goodbye", 

        "thanks": "Acknowledgement",   
        "thank you": "Acknowledgement",

        # Meta queries mapped to Coms.csv intents
        "what is your name": "Bot_Identity",
        "who are you": "Bot_Identity",
        "name?": "Bot_Identity",
        "how can you help": "Assistance",
        "what can you do": "Assistance",
        "assist me": "Assistance",
        "how can you assist me": "Assistance",

        # Scope and Listing Diseases (Scope_Diseases is a new conceptual intent for RAG)
        "what plants do you cover": "Scope", # Assuming Scope is an Intent in Coms.csv for general scope
        "what do you cover": "Scope",
        "what diseases do you know": "Scope_Diseases", # This intent should fetch a list of diseases
        "which diseases do you cover": "Scope_Diseases",
        "leaf diseases you know": "Scope_Diseases",
        "which leaf diseases": "Scope_Diseases",
        "i want to know the leaf diseases you know": "Scope_Diseases",



        # Ensure these keys match the 'Intent' column in disease CSVs
        "symptoms of": "Symptomps", 
        "symptom of": "Symptomps",
        "symptoms": "Symptomps", 
        "symptom": "Symptomps", # Standardized to Symptomps
        "treat": "Treatment_Methods", 
        "treatment": "Treatment_Methods",
        "prevent in future": "Prevention_future",
        "prevention": "Prevention_Guide",
        "cause of": "Cause_Explanation", 
        "causes of": "Cause_Explanation",
        "cause": "Cause_Explanation", 
        "causes": "Cause_Explanation",
        # Add direct matches for intents if users might type them exactly
        "Cause_Explanation": "Cause_Explanation",
        "Symptomps": "Symptomps", 
        "Prevention_Guide": "Prevention_Guide",
        "Saved_Plants": "Saved_Plants", 
        "Impact": "Impact" 
    }

    # Predefined known diseases (lowercase for matching, map to canonical names used in RAG/CSVs)
    known_disease_names_map = {
        "bacterial spot": "Tomato Bacterial Spot",
        "early blight": "Tomato Early Blight",
        "yellow leaf curl": "Tomato Yellow Leaf Curl",
        "tomato yellow leaf curl": "Tomato Yellow Leaf Curl"
        # Add other variations and canonical names as needed
    }
    
    detected_intent = None
    detected_intent_type = "unknown" 
    user_text_lower = user_text_part_for_intent.lower()

    # Sort keys by length to match longer phrases first
    sorted_intent_keys = sorted(intent_keywords.keys(), key=len, reverse=True)

    for key in sorted_intent_keys:
        if key in user_text_lower: # Check if the keyword/phrase is in the user's query
            detected_intent = intent_keywords[key]
            if detected_intent in ["Greetings", "Goodbye", "Acknowledgement"]:
                detected_intent_type = detected_intent 
            elif detected_intent in ["Bot_Identity", "Assistance", "Scope", "Scope_Diseases"]:
                detected_intent_type = "meta_query"
            else: # Assumed to be disease related
                detected_intent_type = "disease_info"
            break 

    # Default RAG search query
    rag_search_query = full_query 

    if detected_intent:
        logger.info(f"Initial detected intent: '{detected_intent}', Type: '{detected_intent_type}'")
        if detected_intent_type in ["Greetings", "Goodbye", "Acknowledgement"] or \
           (detected_intent_type == "meta_query" and detected_intent in ["Bot_Identity", "Assistance", "Scope", "Scope_Diseases"]):
            # For these, the RAG query is the intent name itself, to fetch from Coms.csv or similar
            rag_search_query = detected_intent
        elif detected_intent_type == "disease_info":
            extracted_disease_canonical_name = None
            # Check if disease context is already present
            if disease_context_present: #Extracted disease from image analysis
                extracted_disease_canonical_name = disease
            else:
                # Try to extract disease from text if no image attach with the query
                for disease_key_text, canonical_name in known_disease_names_map.items():
                    if disease_key_text in user_text_lower:
                        extracted_disease_canonical_name = canonical_name
                        break 
            
            if extracted_disease_canonical_name: #Handle for image +text of user
                # Form query like "Symptomps of Tomato Bacterial Spot"
                rag_search_query = f"{detected_intent} of {extracted_disease_canonical_name}" 
            else: # No specific disease identified from text or image for this disease_info intent -> Provide general information
                logger.info(f"Disease intent '{detected_intent}' detected without a specific disease. Forming a general RAG query.")
                natural_intent_term = detected_intent.lower().replace('_', ' ') # e.g., "Symptomps" -> "symptomps"
                rag_search_query = f"general information about {natural_intent_term} for tomato plants"
    else: # No intent detected from keywords
        logger.info("No specific intent keyword matched. Using full_query for RAG.")
        # rag_search_query remains full_query

    logger.info(f"Refined RAG search query to: '{rag_search_query}'")

    #Retrieve Context using Refined Query
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
    #RAG RETRIEVAL BLOCK ENDS HERE 

    #Get History and Construct Final Prompt
    memory = memory_manager.get_session_memory(session_id)
    raw_history_messages = memory.chat_memory.messages
    history_lines = [f"{'User' if msg.type == 'human' else 'AI'}: {msg.content}" for msg in raw_history_messages]
    formatted_history = "\n".join(history_lines)

    system_instruction = """You are AgriBot, an AI assistant specializing in tomato plant diseases.
Your responses MUST strictly follow these rules in the exact order presented.
The final "ASSISTANT ANSWER:" you generate should ONLY contain your direct response to the current "USER QUESTION:". It MUST NOT include labels like "USER QUESTION:", "ASSISTANT ANSWER:", or any other part of the prompt structure.

**Priority Rules (Evaluate in this exact order. Stop at the first rule that applies):**
1.  **Identity Query**:
    *   **Condition**: The 'USER QUESTION' is ONLY and EXACTLY asking for your name (examples: "What is your name?", "Name?", "Who are you?").
    *   **Action**: Your ONLY response MUST be "AgriBot". You MUST COMPLETELY IGNORE all 'CONTEXT DOCUMENTS' and 'CONVERSATION HISTORY' for this specific case. Do not add any other words, greetings, or conversational filler. STOP.

2.  **Greetings & Farewells**:
    *   **Condition**: The 'USER QUESTION' is primarily a common greeting (e.g., 'hello', 'good morning', 'hi') or farewell (e.g., 'goodbye', 'good night', 'bye'), AND Rule #1 did not apply.
    *   **Action**:
        *   **Use Context Path**: If 'CONTEXT DOCUMENTS' (and the content is not "No relevant information found in my knowledge base.") provide a specific response for this greeting/farewell:
            *   **Instruction**: Your ONLY response MUST be the 'CONTEXT DOCUMENTS' as provided. Do not summarize, add to, or omit any part of the 'CONTEXT DOCUMENTS'. Do not add any other conversational text unless it is explicitly part of the 'CONTEXT DOCUMENTS'. STOP.
        *   **Fallback Path (ONLY if Use Context Path conditions were FALSE)**: Otherwise (no specific context found, or context is the default "No relevant information..."):
            *   **Instruction**: Respond with "Hello! How can I assist you today?" for greetings, or a polite, generic farewell (e.g., "Goodbye! Have a great day!") for farewells. Limit your response accordingly. STOP.
            
3.  **Meta Queries (Capabilities, Scope of Help)**:
    *   **Condition**: The 'USER QUESTION' is primarily about your capabilities, what you can help with, or your general scope (e.g., "How can you help me?", "What do you do?", "What plants do you cover?", "What diseases do you know?"), AND Rules #1 and #2 did not apply.
    *   **Action**:
        *   **Use Context Path**: If 'CONTEXT DOCUMENTS' are available AND 'CONTEXT DOCUMENTS' are NOT "No relevant information found in my knowledge base.":
            *   **Instruction**: Base your response on the 'CONTEXT DOCUMENTS'. You can summarize or rephrase the information naturally to answer the user's question about my capabilities or scope. Stick to the information provided in the context. Do not add any other conversational text, greetings, or questions unless they are explicitly part of the 'CONTEXT DOCUMENTS'. **You MUST then STOP processing any further rules.**
        *   **Fallback Path (ONLY if Use Context Path conditions were FALSE)**: If 'CONTEXT DOCUMENTS' are "No relevant information found in my knowledge base." OR if no 'CONTEXT DOCUMENTS' were provided at all:
            *   **Instruction**: Your response MUST be exactly: "I specialize in tomato plant diseases." Do not add any other words, do not ask questions, and do not offer help outside of tomato diseases. **You MUST then STOP processing any further rules.**

4.  **General Scope (Strictly Off-topic)**:
    *   **Condition**: The 'USER QUESTION' is clearly and exclusively not about tomato diseases AND was not covered by Rules #1, #2, or #3.
    *   **Action**: Your ONLY response MUST be "I only specialize in tomato plant health." You MUST COMPLETELY IGNORE all 'CONTEXT DOCUMENTS' and 'CONVERSATION HISTORY' for this specific case. Do not add any other words. **You MUST then STOP processing any further rules.**

**Answering Tomato Disease Questions (Apply only if no Priority Rule above matched):**

5.  **Complex Reasoning / Beyond Direct Retrieval**:
    *   **Condition**: The 'USER QUESTION' asks for complex reasoning, multi-step problem solving, mathematical calculations, or synthesis of information that is not explicitly and directly stated as a whole in the 'CONTEXT DOCUMENTS', even if related to tomato diseases. This includes hypothetical scenarios requiring deductive logic beyond simple fact lookup.
    *   **Action**: Your response MUST be: "I can provide information directly from my knowledge base about tomato diseases. However, I cannot perform complex reasoning or solve multi-step problems that go beyond direct information retrieval from my documents." You MUST COMPLETELY IGNORE 'CONTEXT DOCUMENTS' for generating this specific response. STOP.

6.  **Source of Truth (for direct answers)**:
    *   **Condition**: The 'USER QUESTION' is about tomato diseases, no Priority Rule (#1-#4) was matched, AND Rule #5 did not apply.
    *   **Action**: Base your answers **exclusively** on the information provided in the 'CONTEXT DOCUMENTS' section. Do not use any external knowledge or make assumptions. Proceed to Rule #7 to determine how to respond based on context.

7.  **Responding Based on Context (for Tomato Disease Questions)**:
    *   **Sub-Rule 7a (Sufficient Context)**: If 'CONTEXT DOCUMENTS' (and not "No relevant information found...") provide a direct answer to the 'USER QUESTION' about tomato diseases:
        *   **Action**: Provide the answer using ONLY the information from 'CONTEXT DOCUMENTS'. Apply formatting from Rule #9 if applicable. Do not add any information, questions, or conversational filler not explicitly present in the 'CONTEXT DOCUMENTS'. STOP.
    *   **Sub-Rule 7b (Insufficient Context - No Documents Found)**: If 'CONTEXT DOCUMENTS' are "No relevant information found in my knowledge base.":
        *   **Action**: You MUST state: "I currently do not have information on that specific topic in my knowledge base." Do not ask a question back. STOP.
    *   **Sub-Rule 7c (Insufficient Context - Partial Answer)**: If 'CONTEXT DOCUMENTS' provide some information but do not sufficiently answer all aspects of the 'USER QUESTION' about tomato diseases:
        *   **Action**: First provide the information you *do* have from the context. Then, for the parts of the question you cannot answer from the context, you MUST clearly state what specific information is missing, for example: "...however, I do not have specific details about [the missing aspect] in my knowledge base." Do not ask a question back. STOP.

8.  **Image Analysis Context**: If the 'USER QUESTION' mentions "Image analysis identified:", first acknowledge the identified disease (e.g., "Image analysis identified [Disease Name]."). Then, address the user's specific text query about that disease by strictly following Rules #6 and #7.

9.  **Formatting**: For lists like symptoms, treatments, or prevention steps, use a numbered list if the CONTEXT DOCUMENTS imply a list. Keep answers concise and directly from the context.

10. **No Fabrication**: Under no circumstances should you invent information or provide details not found in the 'CONTEXT DOCUMENTS' when answering questions about tomato diseases. If the 'CONTEXT DOCUMENTS' do not contain the answer, and no other rule dictates a specific response, state that the information is not available (as per Rule 7b or 7c).
"""


    prompt = f"""{system_instruction}

CONVERSATION HISTORY:
{formatted_history if formatted_history else "This is the start of the conversation."}

CONTEXT DOCUMENTS:
{context_for_prompt}

USER QUESTION:
{full_query}

ASSISTANT ANSWER:"""
    logger.debug(f"SENDING PROMPT TO OLLAMA (Session: {session_id}):\n{prompt[:1000]}...") # Log snippet

    ai_response_text = "Sorry, I could not process your request at this moment." # Default fallback
    
    

    #Call LLM
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": config.OLLAMA_MODEL_NAME, 
                "prompt": prompt,
                "stream": False,
                "options": { "temperature": config.OLLAMA_TEMPERATURE } # Use temperature from config
            }
            response = await client.post(config.OLLAMA_API_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()
            ai_response_text = response_data.get("response", "Sorry, I received an empty response from the AI.").strip()

            # This logic attempts to remove any re-generated prompt structure from the AI's response.
            unwanted_structure_start_signals = [
                "\nCONTEXT DOCUMENTS:", 
                "\nUSER QUESTION:", 
                "\nASSISTANT ANSWER:",
                "\nCONVERSATION HISTORY:"
            ]
            
            # Heuristic: Only attempt stripping if the response is reasonably long.
            # Adjust the length threshold as needed.
            if len(ai_response_text) > 75: 
                current_text_to_check = ai_response_text
                earliest_unwanted_index = len(current_text_to_check) # Start with the end

                # Find the earliest occurrence of any unwanted signal *not* at the very beginning
                for signal in unwanted_structure_start_signals:
                    try:
                        # Search for the signal, allowing it to be anywhere *after* the potential start of the actual answer.
                        # A small offset (e.g., 10 characters) helps avoid accidentally cutting off
                        # a legitimate answer that happens to start with a similar phrase (unlikely for these specific signals).
                        search_start_offset = 10 
                        if len(current_text_to_check) > search_start_offset:
                            found_index = current_text_to_check.index(signal, search_start_offset)
                            if found_index < earliest_unwanted_index:
                                earliest_unwanted_index = found_index
                    except ValueError:
                        continue # Signal not found in the latter part of the string
                
                if earliest_unwanted_index < len(current_text_to_check):
                    logger.info(f"Potential unwanted prompt structure detected in AI response. Stripping from index {earliest_unwanted_index}.")
                    ai_response_text = current_text_to_check[:earliest_unwanted_index].strip()

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM service returned error status {e.response.status_code}: {e.response.text}", exc_info=True)
        ai_response_text = f"Sorry, AgriBot's brain had a hiccup (server error {e.response.status_code}). Please try again later."
    except httpx.RequestError as e:
        logger.error(f"Could not connect to LLM service: {e}", exc_info=True)
        ai_response_text = "Sorry, I'm having trouble connecting to my knowledge base. Please check the connection and try again."
    except Exception as e:
        logger.error(f"Unexpected error during LLM call: {e}", exc_info=True)
        ai_response_text = "An unexpected error occurred while I was thinking. Please try again."
    
    
    # The user's message is already added. Now add the AI's response (or error message).
    database.add_message(session_id=session_id, sender='ai', message=ai_response_text)
    

    if not any(err_msg_part in ai_response_text for err_msg_part in [
        "AgriBot's brain had a hiccup", 
        "trouble connecting to my knowledge base", 
        "unexpected error occurred while I was thinking",
        "received an empty response from the AI",
        "could not process your request at this moment"
    ]):
        memory_manager.update_memory(session_id, full_query, ai_response_text)
    else:
        logger.info(f"Skipping memory update for session {session_id} due to error-like AI response.")

    logger.info(f"AI response for session {session_id}: '{ai_response_text[:200]}...'")
    #Return Response
    return ChatResponse(response=ai_response_text, session_id=session_id)