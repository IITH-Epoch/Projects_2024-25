import json
import sys
import os
from flask_cors import CORS  # Ensures proper handling of CORS
import os
import traceback
import requests
from PyPDF2 import PdfReader
import cv2
import numpy as np

# Add your project root to the system path
sys.path.append("D:\\pytorch_projects+tensorflow_projects_3.12\\IITH_GPT\\IITH-GPT\\with_UI_and_memory")

from flask import Flask, request, jsonify
from IITH_GPT.Agentic_code.RAG_LLM import process_query_with_validation
from IITH_GPT.Agentic_code.utils import convert_text_to_markdown_with_gemini,summarize_text

# Import PyMongo and datetime for MongoDB operations
from pymongo import MongoClient
from datetime import datetime

# For image processing
from PIL import Image
import io
import pytesseract

# For PDF processing
from PyPDF2 import PdfReader

# Setup MongoDB client.
MONGODB_URI = "MONGODB KEY"
client = MongoClient(MONGODB_URI)
db = client["test"]
chat_collection = db["chatHistory"]

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        # Validate required fields
        if not data or 'query' not in data:
            return jsonify({"error": "Invalid request: 'query' field is required"}), 400

        user_query = data['query']
        user_email = data.get("email", "")
        if not user_email:
            return jsonify({"error": "Email is required."}), 400
        
        # Get the bot response using your function
        bot_response = process_query_with_validation(query=user_query, user_email=user_email)
        print("Bot response:", bot_response)

        # Save chat history to MongoDB
        chat_collection.insert_one({
            "query": user_query,
            "response": bot_response,
            "timestamp": datetime.utcnow(),
            "email": user_email
        })
        
        # Keep only the last 10 messages for this user
        total_docs = chat_collection.count_documents({"email": user_email})
        if total_docs > 10:
            docs_to_delete = chat_collection.find({"email": user_email}).sort("timestamp", 1).limit(total_docs - 10)
            ids_to_delete = [doc["_id"] for doc in docs_to_delete]
            chat_collection.delete_many({"_id": {"$in": ids_to_delete}})
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

def preprocess_image(image):
    """
    Preprocesses the image to improve OCR accuracy.
    Steps:
    1. Convert to grayscale
    2. Resize the image for better readability
    3. Apply median blur to reduce noise
    4. Apply adaptive thresholding for better contrast
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to enhance small text (scale factor can be adjusted)
    scale_factor = 2
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Apply median blur to reduce noise
    gray = cv2.medianBlur(gray, 3)
    
    # Use adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Ensure an image file was provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Read the image file into memory
        image_bytes = image_file.read()
        
        # Convert the image bytes into a NumPy array and decode with OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_cv is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Preprocess the image using OpenCV for better OCR accuracy
        processed_image = preprocess_image(image_cv)
        
        # Set custom Tesseract configuration options
        config = "--oem 3 --psm 6"
        
        # Perform OCR on the preprocessed image
        ocr_text = pytesseract.image_to_string(processed_image, config=config)
        print("OCR extracted text:", ocr_text)
        
        # Convert the OCR text to Markdown using your Gemini conversion function
        markdown_text = convert_text_to_markdown_with_gemini(ocr_text)
        print("Converted Markdown text:", markdown_text)
        
        # Return the Markdown output
        return jsonify({"markdown": markdown_text})
    
    except Exception as e:
        print("Error processing image:", e)
        return jsonify({"error": "An error occurred while processing the image."}), 500

@app.route('/upload-document', methods=['POST'])
def upload_document():
    try:
        # Check if a document file was provided
        if 'document' not in request.files:
            return jsonify({"error": "No document file provided"}), 400

        document_file = request.files['document']
        if document_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = document_file.filename
        ext = os.path.splitext(filename)[1].lower()
        document_text = ""

        # Extract text from a PDF or TXT file
        if ext == ".pdf":
            pdf_reader = PdfReader(document_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    document_text += text
        elif ext == ".txt":
            document_text = document_file.read().decode('utf-8')
        else:
            return jsonify({"error": "Unsupported document format. Please upload a PDF or TXT file."}), 400

        # Optional: Retrieve email from form data (if needed)
        email = request.form.get("email", "anonymous@example.com")
        print("Extracted document text:", document_text)  # Debug log

        # Generate the summary using the Gemini API
        summary = summarize_text(document_text)
        print("Generated summary:", summary)  # Debug log

        # Convert the summary to Markdown using Gemini
        markdown_text = convert_text_to_markdown_with_gemini(summary)
        print("Converted Markdown text:", markdown_text)  # Debug log

        # Return the Markdown output
        return jsonify({"markdown": markdown_text})
    except Exception as e:
        print("Error processing document:", e)
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing the document."}), 500

if __name__ == '__main__':
    # Running the application on all available interfaces (host='0.0.0.0') and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
