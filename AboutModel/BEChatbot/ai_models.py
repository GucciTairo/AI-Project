# ai_models.py
import logging #Logging Message(debug,info,warning)
import os
import io
import numpy as np
from PIL import Image

#Converting text into numerical vector(embeddings) - power for RAG search
from sentence_transformers import SentenceTransformer
import keras 
from keras import mixed_precision as keras_mixed_precision 
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as tf_keras_vgg16_preprocess_input

import config

#Globals for loaded models
embedding_model = None
image_model = None

#APPLY KERAS 3 MIXED PRECISION POLICY GLOBALLY
try:
    module_logger = logging.getLogger(__name__ + ".startup") #name = ai_models.py, logging the mixed policy setup
    policy = keras_mixed_precision.Policy('mixed_float16') # Use keras_mixed_precision
    keras_mixed_precision.set_global_policy(policy)         # Use keras_mixed_precision
    module_logger.info(f"Global Keras (standalone) mixed precision policy set to: {keras_mixed_precision.global_policy().name}")
except Exception as e:
    module_logger = logging.getLogger(__name__ + ".startup")
    module_logger.warning(f"Could not enable Keras global mixed precision: {e}")


#Model Loading Functions
def load_embedding_model():
    global embedding_model #define embedding_model as global variable
    if embedding_model is None: #Checking for if embedding model is already loaded
        try:
            logger = logging.getLogger(__name__) # Get logger for this module
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}") #Get logger.info of loading embedding model -> ensure the model name load correctly
            embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME) # Load the embedding model using SentenceTransformer: all-MiniLM-L6-v2
            logger.info("Loaded successfully.")
        except Exception as e:
            logger.error(f"Fail INFO on this loggerled to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return embedding_model

def load_image_model():
    global image_model #Define global variable image_model
    logger = logging.getLogger(__name__) # Get logger for this module
    if image_model is None: #If not having any module name
        # Ensure config.IMAGE_MODEL_PATH points to the .keras file
        if not os.path.exists(config.IMAGE_MODEL_PATH): # Its will move to config file and load from this model path
            logger.warning(f"Image model path not found: {config.IMAGE_MODEL_PATH}. Image analysis disabled.")
            return None
        try:
            logger.info(f"Loading Keras image model from: {config.IMAGE_MODEL_PATH} using standalone keras.models.load_model...")
            # CRITICAL: Use the standalone keras.models.load_model
            image_model = keras.models.load_model(config.IMAGE_MODEL_PATH)
            logger.info("Keras image model loaded successfully using standalone Keras.")
            # Optionally log model summary to confirm it's the correct architecture
            if logger.isEnabledFor(logging.DEBUG) and image_model:
                image_model.summary(print_fn=logger.debug)
        except Exception as e:
            logger.error(f"Failed to load Keras image model using standalone Keras from {config.IMAGE_MODEL_PATH}: {e}", exc_info=True)
            image_model = None
    return image_model

#Image Processing
#Object img: Image.Image input
def preprocess_image(img: Image.Image) -> np.ndarray | None: #Expect 1 arguments
    logger = logging.getLogger(__name__) 
    try: #Try except block for error handling -> The main code inside try block. If any error occurs during running -> jump to except block instead of crashing
        if img.mode != 'RGB': #If image is not RGB -> convert it to RGB
            img = img.convert('RGB')
        img = img.resize(config.IMAGE_TARGET_SIZE) #Resize image to target size inside file config
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array.astype('float32') # VGG16 preprocess_input usually expects float32

        # Use VGG16 specific preprocessing from tensorflow.keras.applications
        img_array = tf_keras_vgg16_preprocess_input(img_array)
        logger.debug("Applied VGG16 specific preprocess_input (from tf.keras).")
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

#Predict image class predict the class of disease -> with confidence scores
def predict_image_class(img: Image.Image) -> tuple[str | None, float | None]: #Return none if prediction fails
    logger = logging.getLogger(__name__) 
    if image_model is None: #Check for image model is loaded
        logger.warning("Predict image class called but image_model is not loaded.")
        return None, None
    processed_image = preprocess_image(img) #Call the function preprocess_image to convert raw image to the expected format for VGG16
    if processed_image is None: #Check for image process call and convert to expected
        logger.warning("Image preprocessing failed, cannot predict.")
        return None, None
    try:
        logger.debug(f"Predicting with image model. Input shape: {processed_image.shape}, dtype: {processed_image.dtype}") #Check for the debug of image model. If the input shape and dtype format does not match the requirements
        prediction = image_model.predict(processed_image, verbose=0) #Calling model to predict the image already processed, using verbose = 0 to run silently
        logger.debug(f"Raw prediction output: {prediction}") #Check for the raw prediction output
        predicted_class_index = np.argmax(prediction[0]) #With 3 classes of model training -> Which having highest score will be shown out, correspond to the predicted class
        confidence = float(np.max(prediction[0])) #Convert to standard python float
        if 0 <= predicted_class_index < len(config.IMAGE_CLASS_NAMES): #Check for the predicted class index is within the bounds of the class names list
             predicted_class_name = config.IMAGE_CLASS_NAMES[predicted_class_index] #Get the predicted class name from the config.IMAGE_CLASS_NAMES list
             logger.info(f"Image prediction successful: {predicted_class_name} ({confidence:.4f})")
             return predicted_class_name, confidence
        else:
             logger.warning(f"Predicted class index {predicted_class_index} is out of bounds for class names list (len: {len(config.IMAGE_CLASS_NAMES)}).")
             return "Unknown Class", confidence
    except Exception as e:
        logger.error(f"Error during image prediction with Keras model: {e}", exc_info=True)
        return None, None
#This function using to convert text into numerical vector(embeddings) - power for RAG search
def get_embedding(text: str) -> list[float] | None:
    logger = logging.getLogger(__name__)
    if embedding_model is None:
        logger.error("Get embedding called but embedding_model (global) is not loaded.")
        return None
    try:
        logger.debug(f"Generating embedding for text (first 50 chars): '{text[:50]}...'")
        embedding_array = embedding_model.encode([text]) #Call the Sentence Transformer model to encode the text into an embedding vector
        #the input [text] is wrapped in a list
        embedding_list = embedding_array[0].tolist() #Convert Numpy the array embedding into a standard Python list [0.123,0.456,...,0.789] within 2D array 
        logger.debug(f"Embedding generated, dimension: {len(embedding_list)}")
        return embedding_list
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
        return None