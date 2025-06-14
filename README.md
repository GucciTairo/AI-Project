## 1. ChromaDB

### a. ChromaDB usage in our project

- **Vector Database:** Manages embeddings.  
- **Local Storage:** Runs easily on our machine.  
- **Efficient Search:** Enables fast semantic similarity searches.  
- **Powers RAG:** Provides the specific context to the LLM for grounded answers.  
- **Focuses LLM:** Helps the generative AI stay on topic.

![Base on this images](2025-05-31-23-48-40.png)

[ChromaDB, Embedding and RAG](https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/#:~:text=The%20LangChain%20framework%20allows%20you%20to%20build%20a,to%20LangChain%3F%20Start%20with%20this%20introductory%20post%20first)  

### b. ChromaDB Collections

Collections in ChromaDB are analogous to tables in traditional databases.  
They serve as containers to organize and store embeddings along with their associated data and metadata.

![Chroma Collections](2025-05-31-23-27-50.png)

[Comprehensive Beginner’s Guide to ChromaDB](https://medium.com/@syeedmdtalha/a-comprehensive-beginners-guide-to-chromadb-eb2fa22ee22f)  

Each collection is created with a unique name.  
We can add, update, query, or delete embeddings from a collection.

### c. Embedding model with RAG, store on ChromaDB

1. Our Knowledge Base (CSVs -> ChromaDB):

    - We take the text from the Response column (and potentially other relevant columns) of our CSV files.
    - For each piece of text (each "document" we want to store), we use the embedding model (e.g., all-MiniLM-L6-v2) to convert that text into a vector (a list of numbers).
    - These vectors, along with their original text and any metadata, are stored in your ChromaDB vector database. This happens during preprocessing step.

![Embedding Model usage](2025-06-03-12-42-28.png)

2. User Asks a Question (Runtime):

    - The user types a query like "What are symptoms of bacterial spot?".
    - Our website/application takes this query string.
    - It uses the same embedding model (all-MiniLM-L6-v2) to convert the user's query text into a vector.

3. Similarity Search in ChromaDB:

    - The vector representing the user's query is sent to ChromaDB.
    - It finds the document vectors that are "closest" (most similar) to the query vector.

4. Retrieve Documents:

    - ChromaDB returns the original text documents associated with those most similar vectors.

5. Augment LLM Prompt:

    - These retrieved text documents are then inserted into the prompt send to the LLM (Phi-2), along with your instructions and the user's original query.

[Embedding Models for Effective Retrieval-Augmented Generation](https://medium.com/bright-ai/choosing-the-right-embedding-for-rag-in-generative-ai-applications-8cf5b36472e1)  

## 2. Building Chatbot

### a. Building for GenAI + TraAI
    #### GenAI
    - Config.py:
        - Contains configuration settings for the chatbot.
        - [Includes API keys, model names, and other parameters](https://dev.to/jayantaadhikary/using-the-ollama-api-to-run-llms-and-generate-responses-locally-18b7) 
        
        - Other settings:  
        N_RESULTS = 3
            - When a user asks a question,our application converts that question into an embedding (a numerical vector).  
            - This query embedding is then sent to ChromaDB.  
            - ChromaDB compares the query embedding to all the document embeddings stored in your tomato_chatbot_knowledge collection.  
            - It finds the documents whose embeddings are most semantically similar (closest in vector space) to the query embedding.  
            - N_RESULTS = 3 tells ChromaDB to return the top 3 most similar documents.  
        MEMORY_K = 6
            - Remember last 6 response in chathistory.db  
    
    - AI_models.py:  
        - With 5 main function to handle for GenAI and TraAI in chatbot  
            - def load_embedding_model(): load the embedding model (all-MiniLM-L6-v2 from config.py) for converting text to vectors.  
            - def load_image_model(): load the image model (Tomato VGG16) for predict diseases using Keras,..  
            - def preprocesss_image(): handle image preprocessing for the image model with required 224x224 and float32 to predict the class name  
            - def predict_image_class(): calling image_model.predict() to detect diseases  
            - def get_embedding(): convert text to embedding using the loaded embedding model: https://www.geeksforgeeks.org/nlp/text-embeddings-using-openai/  
    
    - chatbot_app.py:  We using API endpoint to handle the chatbot's functionality.  
        - 
    
    - database.py: Using SQL to store our chat history of user and AI responses in chathistory.db  
        - def init_db(): initialize the SQLite database and create the chat_history table if it doesn't exist.  
        - def add_message(): add a new message to the chat history table.  
    
    - memory_manager.py: using for managing the conversation history(memory) for different chat sessions. We use LangChain's ConversationBufferWindowMemory to keep track of the last N messages in the conversation.  
        - Follow this instruction to code for ConversationBufferWindowMemory: https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html, https://github.com/langchain-ai/langchain/discussions/10075
        - def get_session_memory():  
            - This function retrieves the conversation memory for a specific session ID.  
            - If the session does not exist, it creates a new memory instance with a specified window size (MEMORY_K).  
        - def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:  
            - This function retrieves the conversation memory for a specific session ID.  
            - If the session does not exist, it creates a new memory instance with a specified window size (MEMORY_K).  
        - def format_chat_history_for_prompt(): 
            - This function formats the chat history for the prompt sent to the LLM.  
            - It retrieves the session memory and formats it into a string that includes both user and AI messages.  
            - The formatted chat history is then returned as a string, which can be used in the prompt for the LLM.  
        - def update_memory:  
            - This function updates the conversation memory with a new user message and the AI's response.  
            - It adds the user message to the session memory and then appends the AI's response to the chat history.  
            - The updated chat history is returned as a string, which can be used in subsequent prompts.  

    #### TraAI  
    All on report here: https://docs.google.com/document/d/1n5ivWsCn20Qrk995mFfOthLH91hnT3iMn0nq6cgZYI0/edit?pli=1&tab=t.0  
    After training model with file .keras. Our team use it and run with API for the model  

    #### Docker:  
        - Using file requirement.txt to install all the required packages for the chatbot.  
        - Using file Dockerfile to build the docker image for the chatbot.  
    

    #### DataSet:  
        - CSV files containing the knowledge base for the chatbot. Our team building it from gathering data from Internet
        - Images for training the image model to detect diseases. You guy can get it from here: https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources  
        - Here is our dataset after modifying for 224x224 and uint8 + with the dataset for RAG of chatbot: 
    
    #### Building for web UI:
        - html, script,style
