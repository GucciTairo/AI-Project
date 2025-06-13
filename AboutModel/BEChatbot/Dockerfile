# Dockerfile

#Base Image
    FROM python:3.11-slim

    #Environment Variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    # Keep common install location in PATH, pip sometimes uses it too
    ENV LOCAL_BIN_DIR="/root/.local/bin"
    ENV PATH="${LOCAL_BIN_DIR}:${PATH}"
    
    # Set Working Directory
    WORKDIR /app
    
    # Install Dependencies using pip
    # Copy requirements first for caching
    COPY requirements.txt .
    
    # Install using pip, AND verify uvicorn immediately after
    RUN echo "Running pip install" \
        && pip install --no-cache-dir -r requirements.txt \
        && echo "Verifying uvicorn installation" \
        && which uvicorn \
        && python -m uvicorn --version \
        && echo "Installation verification successful."
    
    #Copy Application Code
    COPY . .
    
    #Expose Port
    EXPOSE 8000
    
    #Run Application
    CMD ["uvicorn", "chatbot_app:app", "--host", "0.0.0.0", "--port", "8000"]