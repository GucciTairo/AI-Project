document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const sendButton = document.getElementById('sendButton');
    const statusDiv = document.getElementById('status');
    const clearImageButton = document.getElementById('clearImageButton');

    const CHAT_API_URL = 'http://localhost:8000/chat';
    const IMAGE_API_URL = 'http://localhost:8000/analyze_image';

    let currentSessionId = `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    console.log(`Session ID: ${currentSessionId}`);
    let selectedImageFile = null;
    // globalAnalysisResult is used to hold the latest image analysis result,
    // primarily to know if an error occurred during analysis.
    let globalAnalysisResult = null; 

    // Helper to add a message (text OR image) to the chatbox
    function displayMessage(content, sender = 'ai', isImage = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(sender === 'user' ? 'user-message' : 'ai-message');

        if (isImage && sender === 'user') {
            // Content here is expected to be a data URL (from FileReader)
            const imgElement = document.createElement('img');
            imgElement.src = content; 
            imgElement.alt = "User uploaded image";
            imgElement.classList.add('chat-image');
            messageElement.appendChild(imgElement);
        } else if (sender === 'ai') {
            // For AI messages, use innerHTML to render <br> tags from newlines.
            // Be aware that if 'content' could contain malicious HTML from the AI (unlikely with controlled LLMs),
            // this could be a security risk. For typical chatbot text, it's fine.
            messageElement.innerHTML = content.replace(/\n/g, '<br>');
        } else {
            // For user text messages
            messageElement.textContent = content;
        }

        chatbox.appendChild(messageElement);
        chatbox.scrollTop = chatbox.scrollHeight; // Auto-scroll to the latest message
        console.log(`Displayed ${sender} message (isImage: ${isImage}): ${typeof content === 'string' ? content.substring(0,100) : '[Image Data]'}`);
    }

    // Helper to set status messages and UI element states
    function setStatus(message = '', isLoading = false, isError = false) {
        console.log(`Setting status: "${message}", isLoading: ${isLoading}, isError: ${isError}`);
        statusDiv.textContent = message;
        sendButton.disabled = isLoading;
        userInput.disabled = isLoading;
        imageInput.disabled = isLoading;
        clearImageButton.disabled = isLoading; // Also disable clear image button during loading

        if (isLoading) {
            statusDiv.style.color = "#555"; // Neutral color for loading messages
        } else if (isError) {
            statusDiv.style.color = "#dc3545"; // Red for error messages
        } else if (message) { // Not loading, not an error, but has a message (implies success or info)
            statusDiv.style.color = "#28a745"; // Green for success/info messages
        } else { // Not loading, not an error, no message (idle state)
            statusDiv.style.color = "#888"; // Default/idle color
        }
    }
    
    // Helper to clear all inputs and the image preview
    function clearAllInputsAndPreview() {
        console.log("Clearing inputs and preview.");
        userInput.value = '';
        imageInput.value = ''; // Resets the file input
        if (imagePreview) {
            imagePreview.src = '#'; // Or use ''
            imagePreview.style.display = 'none';
        }
        if (clearImageButton) {
            clearImageButton.style.display = 'none';
        }
        selectedImageFile = null;
        globalAnalysisResult = null; // Clear analysis result as well
    }

    // API call to analyze the selected image
    async function analyzeImageAPI(imageFile) {
        setStatus('Analyzing image...', true, false);
        const formData = new FormData();
        formData.append('file', imageFile, imageFile.name); // Ensure filename is passed
        try {
            const response = await fetch(IMAGE_API_URL, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `Image analysis failed with HTTP status ${response.status}` }));
                throw new Error(errorData.error || `Image analysis failed (Status: ${response.status})`);
            }
            const result = await response.json();
            console.log("Image Analysis Result (raw):", result);
            globalAnalysisResult = result; // Store for potential later use or checking errors
            return result;
        } catch (error) {
            console.error('Error in analyzeImageAPI:', error);
            // Set status with error, but allow further processing (e.g., sending text query)
            setStatus(`Image Analysis Error: ${error.message}`, false, true); 
            globalAnalysisResult = { error: error.message }; // Store error state
            return globalAnalysisResult; // Return error object
        }
    }

    // API call to send the chat query
    async function sendChatQueryAPI(queryText, sessionId) {
        // Avoid overwriting "Image analyzed" status if it was just set
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
                const errorData = await response.json().catch(() => ({ detail: `Chat request failed with HTTP status ${response.status}` }));
                throw new Error(errorData.detail || `Chat request failed (Status: ${response.status})`);
            }
            const result = await response.json();
            console.log("Chat Response (raw):", result);

            if (result.session_id) { currentSessionId = result.session_id; } // Update session ID if backend provides a new one

            // Display the AI's response. result.response should contain the clean AI message.
            displayMessage(result.response || "AI did not provide a text response.", 'ai');
            setStatus(''); // Clear status after successful send and display
        } catch (error) {
            console.error('Full error object in sendChatQueryAPI:', error);
            setStatus(`Chat Error: ${error.message}`, false, true);
            displayMessage(`AI Error: ${error.message}`, 'ai'); // Display error in chat as an AI message
        } finally {
            // Ensure UI is re-enabled, preserving error/status message if one was set by catch
            // The setStatus in catch block would have set isError=true.
            // If successful, setStatus('') above would have cleared the message.
            // This final setStatus ensures isLoading is false.
            const currentStatusMessage = statusDiv.textContent;
            const isCurrentlyError = statusDiv.style.color === 'rgb(220, 53, 69)'; // Check for red color
            setStatus(currentStatusMessage, false, isCurrentlyError);
        }
    }

    // Event Listener for the Send Button
    sendButton.addEventListener('click', async () => {
        const userTextQuery = userInput.value.trim();
        const imageFileToProcess = selectedImageFile; // Use the file selected via input

        if (!userTextQuery && !imageFileToProcess) {
            setStatus('Please enter a query or select an image.', false, true);
            return;
        }

        // Display user's text query in chat
        if (userTextQuery) {
            displayMessage(userTextQuery, 'user');
        }
        // Display user's selected image in chat
        if (imageFileToProcess && imagePreview.src && imagePreview.src !== '#' && imagePreview.style.display === 'block') {
            displayMessage(imagePreview.src, 'user', true); // imagePreview.src is the data URL
        }
        
        setStatus('Processing...', true, false);

        let combinedQueryText = userTextQuery;
        let imageAnalysisSuccessful = false;
        let detectedDiseaseFromImage = null;

        if (imageFileToProcess) {
            const analysisData = await analyzeImageAPI(imageFileToProcess); 
            
            if (analysisData && analysisData.disease && !analysisData.error) {
                setStatus('Image analyzed. Preparing chat query...', true, false); // Intermediate status
                detectedDiseaseFromImage = analysisData.disease;
                const confidenceScore = analysisData.confidence ? (analysisData.confidence * 100).toFixed(1) : "N/A";
                const analysisText = `Image analysis identified: ${analysisData.disease} (Confidence: ${confidenceScore}%).`;
                
                if (userTextQuery) {
                    combinedQueryText = `${analysisText} User query: ${userTextQuery}`;
                } else {
                    combinedQueryText = `${analysisText} User query: (Regarding the analyzed image)`;
                }
                imageAnalysisSuccessful = true;
            } else {
                // Image analysis failed or returned an error. Status already set by analyzeImageAPI.
                // If there's no text query, and image analysis failed, we can't proceed.
                if (!userTextQuery) {
                    // analyzeImageAPI already set an error status. We just ensure inputs are cleared.
                    clearAllInputsAndPreview(); 
                    // setStatus in analyzeImageAPI's catch block handles enabling buttons.
                    return; 
                }
                // If there's a text query, proceed with it despite image analysis failure.
                combinedQueryText = userTextQuery; 
            }
        }
        
        // Send the query to the chat API
        if (combinedQueryText) {
            await sendChatQueryAPI(combinedQueryText, currentSessionId);
        } else if (imageFileToProcess && !imageAnalysisSuccessful) {
            // This case means an image was selected, analysis failed, and there was no text query.
            // Should have been caught above, but as a fallback.
            setStatus('Image analysis failed and no text query was provided.', false, true);
        } else if (!imageFileToProcess && !userTextQuery){
             // Should be caught by the initial check, but as a safeguard.
             setStatus('Nothing to send to AI.', false, true);
        }
        
        // Clear inputs for the next message, but not the status if an error occurred.
        const currentStatus = statusDiv.textContent;
        const isErrorState = statusDiv.style.color === 'rgb(220, 53, 69)';
        
        clearAllInputsAndPreview();
        userInput.focus();

        // Restore status if it was important (e.g., an error message)
        if (currentStatus && isErrorState) {
            setStatus(currentStatus, false, true);
        } else if (currentStatus && !isErrorState && statusDiv.style.color === 'rgb(40, 167, 69)') { // Success color
             setStatus(currentStatus, false, false); // Keep success message briefly
        }
    });

    // Event Listener for Enter key in the text input
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { // Send on Enter, allow Shift+Enter for newlines
            e.preventDefault(); // Prevent default Enter behavior (e.g., form submission)
            sendButton.click();
        }
    });

    // Event Listener for image selection
    imageInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files && files[0]) {
            selectedImageFile = files[0];
            globalAnalysisResult = null; // Reset previous analysis result
            const reader = new FileReader();
            reader.onload = (e) => {
                if (imagePreview) {
                    imagePreview.src = e.target.result; // This is the data URL
                    imagePreview.style.display = 'block';
                }
                if (clearImageButton) {
                    clearImageButton.style.display = 'inline-block';
                }
            }
            reader.readAsDataURL(selectedImageFile);
            setStatus(''); // Clear any previous status message
        }
    });

    // Event Listener for the clear image button
    if (clearImageButton) {
        clearImageButton.addEventListener('click', () => {
            clearAllInputsAndPreview();
            setStatus(''); // Clear status when image is cleared
        });
    }
});