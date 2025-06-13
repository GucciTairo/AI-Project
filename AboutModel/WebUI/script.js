document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview'); // We have this for the input area
    const sendButton = document.getElementById('sendButton');
    const statusDiv = document.getElementById('status');
    const clearImageButton = document.getElementById('clearImageButton');

    const CHAT_API_URL = 'http://localhost:8000/chat';
    const IMAGE_API_URL = 'http://localhost:8000/analyze_image';

    let currentSessionId = `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    console.log(`Session ID: ${currentSessionId}`);
    let selectedImageFile = null;
    let globalAnalysisResult = null;

    // Helper to add a message (text OR image) to the chatbox
    function displayMessage(content, sender = 'ai', isImage = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(sender === 'user' ? 'user-message' : 'ai-message');

        if (isImage && sender === 'user') {
            // content here is expected to be a data URL (from FileReader) or an object URL
            const imgElement = document.createElement('img');
            imgElement.src = content; // Set the src to the image data URL
            imgElement.alt = "User uploaded image";
            imgElement.classList.add('chat-image'); // Add class for styling
            messageElement.appendChild(imgElement);
        } else if (sender === 'ai') {
            messageElement.innerHTML = content.replace(/\n/g, '<br>');
        } else {
            messageElement.textContent = content;
        }

        chatbox.appendChild(messageElement);
        chatbox.scrollTop = chatbox.scrollHeight;
        console.log(`Displayed ${sender} message (isImage: ${isImage}): ${typeof content === 'string' ? content.substring(0,100) : '[Image Data]'}`);
    }

    // (setStatus, clearAllInputsAndPreview, analyzeImageAPI, sendChatQueryAPI functions remain the same as the previous version)
    // Helper to set status and button state (same as before)
    function setStatus(message = '', isLoading = false, isError = false) {
        console.log(`Setting status: "${message}", isLoading: ${isLoading}, isError: ${isError}`);
        statusDiv.textContent = message;
        sendButton.disabled = isLoading;
        userInput.disabled = isLoading;
        imageInput.disabled = isLoading;

        if (isLoading) {
            statusDiv.style.color = "#555";
        } else if (isError) {
            statusDiv.style.color = "#dc3545";
        } else {
            if (!message && !isError) {
                 statusDiv.style.color = "#888";
            } else if (message) {
                 statusDiv.style.color = isError ? "#dc3545" : "#28a745";
            }
        }
    }
    
    // Helper to clear inputs and preview (same as before)
    function clearAllInputsAndPreview() {
        console.log("Clearing inputs and preview.");
        userInput.value = '';
        imageInput.value = '';
        imagePreview.src = '#';
        imagePreview.style.display = 'none';
        clearImageButton.style.display = 'none';
        selectedImageFile = null;
        globalAnalysisResult = null;
    }

    // API call to analyze image (same as before)
    async function analyzeImageAPI(imageFile) {
        setStatus('Analyzing image...', true, false);
        const formData = new FormData();
        formData.append('file', imageFile, imageFile.name);
        try {
            const response = await fetch(IMAGE_API_URL, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
                throw new Error(errorData.error || `Image analysis failed (Status: ${response.status})`);
            }
            const result = await response.json();
            console.log("Image Analysis Result (raw):", result);
            globalAnalysisResult = result;
            return result;
        } catch (error) {
            console.error('Error in analyzeImageAPI:', error);
            setStatus(`Image Analysis Error: ${error.message}`, false, true);
            globalAnalysisResult = { error: error.message };
            return globalAnalysisResult;
        }
    }

    // API call to send chat query (same as before)
    async function sendChatQueryAPI(queryText, sessionId) {
        if (!statusDiv.textContent.includes("Image analyzed")) {
            setStatus('Sending to AI...', true, false);
        }
        const payload = { query: queryText, session_id: sessionId };
        console.log("Payload for /chat:", JSON.stringify(payload));
        try {
            const response = await fetch(CHAT_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP error ${response.status}` }));
                throw new Error(errorData.detail || `Chat request failed (Status: ${response.status})`);
            }
            const result = await response.json();
            console.log("Chat Response (raw):", result);
            if (result.session_id) { currentSessionId = result.session_id; }
            console.log("Attempting to display AI message from /chat response:", result.response);
            displayMessage(result.response || "AI did not provide a text response.", 'ai');
            setStatus('');
        } catch (error) {
            console.error('Full error object in sendChatQueryAPI:', error);
            setStatus(`Chat Error: ${error.message}`, false, true);
            displayMessage(`AI Error: ${error.message}`, 'ai');
        } finally {
            setStatus(statusDiv.textContent, false, statusDiv.style.color === 'rgb(220, 53, 69)');
        }
    }


    // Event Listeners
    sendButton.addEventListener('click', async () => {
        const userTextQuery = userInput.value.trim();
        const imageFileToProcess = selectedImageFile;

        if (!userTextQuery && !imageFileToProcess) {
            setStatus('Please enter a query or select an image.', false, true);
            return;
        }

        //Display user input in chat 
        if (userTextQuery) {
            displayMessage(userTextQuery, 'user');
        }
        if (imageFileToProcess) {
            // Display the image preview data URL in the chat directly
            // imagePreview.src contains the data URL from FileReader
            if (imagePreview.src && imagePreview.src !== '#' && imagePreview.style.display === 'block') {
                displayMessage(imagePreview.src, 'user', true); // Pass true for isImage
            } else {
                // Fallback if preview src isn't set but file exists (should not happen with current logic)
                displayMessage(`(Processing image: ${imageFileToProcess.name})`, 'user');
            }
        }
        //End display user input
        
        setStatus('Processing...', true, false);

        let combinedQueryText = userTextQuery;
        let imageAnalysisSuccessful = false;
        // Store the disease result from image analysis temporarily for chat query construction.
        let detectedDiseaseFromImage = null;


        if (imageFileToProcess) {
            const analysisData = await analyzeImageAPI(imageFileToProcess); 
            
            if (analysisData && analysisData.disease && !analysisData.error) {
                setStatus('Image analyzed. Preparing chat query...', true, false);
                detectedDiseaseFromImage = analysisData.disease; // Store for constructing query
                const analysisText = `Image analysis identified: ${analysisData.disease} (Confidence: ${(analysisData.confidence * 100).toFixed(1)}%).`;
                combinedQueryText = `${analysisText} User query: ${userTextQuery ? userTextQuery : '(Regarding the analyzed image)'}`;
                imageAnalysisSuccessful = true;
            } else {
                const errorText = (analysisData && analysisData.error) ? analysisData.error : "Image analysis failed.";
                setStatus(`Image Analysis Error: ${errorText}. Proceeding with text query if available.`, false, true);
                // If image analysis fails, still proceed with text query if any
                combinedQueryText = userTextQuery; 
                 if (!userTextQuery) {
                    clearAllInputsAndPreview();
                    // setStatus ensures button is re-enabled by isLoading=false
                    setStatus(`Error: ${errorText}`, false, true); 
                    return; 
                }
            }
        }
        
        if (combinedQueryText) {
            await sendChatQueryAPI(combinedQueryText, currentSessionId);
        } else if (imageFileToProcess && imageAnalysisSuccessful && detectedDiseaseFromImage) {
            // This case: only image was provided, and it was analyzed successfully.
            combinedQueryText = `The image analysis identified: ${detectedDiseaseFromImage}. Please tell me more about this disease, like its symptoms or treatment.`;
            await sendChatQueryAPI(combinedQueryText, currentSessionId);
        } else if (!imageFileToProcess && !userTextQuery){
             setStatus('Nothing to send to AI.', false, true);
        }
        
        clearAllInputsAndPreview();
        userInput.focus();
    });

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });

    imageInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files && files[0]) {
            selectedImageFile = files[0];
            globalAnalysisResult = null; 
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result; // This data URL will be used for display
                imagePreview.style.display = 'block';
                clearImageButton.style.display = 'inline-block';
            }
            reader.readAsDataURL(selectedImageFile);
            setStatus('');
        }
    });

    clearImageButton.addEventListener('click', () => {
        clearAllInputsAndPreview();
    });
});