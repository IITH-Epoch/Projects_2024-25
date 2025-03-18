# Meet Snap

---

This project provides a comprehensive pipeline for processing video files to extract keyframes and generate detailed summaries. It utilizes GPU acceleration for video processing, AI-based captions and summaries.

## Features:

1. **Audio Extraction**: Extracts audio from video using FFmpeg with GPU acceleration.
2. **Key Frame Detection:** Identifies key frames using dynamic thresholds based on frame differences.
3. **Summary Generation:** Generates a detailed summaries of selected key frames and audio transcription.

## **Key Functions**

### **1. Audio Extraction (`extract_audio`)**

Extracts audio from the input video file and generates a transcript using Google Generative AI Gemini through API.

### **2. Frame Capture (`capture_frames`)**

Captures frames from the video at a rate of one frame per second using FFmpeg with GPU acceleration.

### **3. Frame Selection (`important_frames`)**

Finds Distinct frames based on dynamic thresholds calculated from frame differences.

### **4. Caption Generation (`summary_images`)**

Generates detailed captions for selected Distinct frames using AI models, and filters the unimportant, repetitive and low information frames.

### **5. PDF Report Generation (`make_pdf`)**

Takes audio transcripts and detailed summary of keyframes (which we got after filtering out distinct frames) and store them in a PDF file and generates a very detailed summary of the PDF file through Gemini.

## Pipeline

1. After giving the image as input, we will audio using FFmpeg and extract one frame per second from the video using FFmpeg. And the transcript of the audio using Gemini.
2. Then we apply dynamic thresholding to filter out repetitive and similar frames. This is the first filtering of frames.
3. Then we generate captions for each frame using Gemini. From the captions we got, we will filter out the similar, low information and repetitive frames using Gemini. This is second filtering of frames. Now we have Key frames of Video and audio transcript.
4. Now we generate detailed summaries of each key frames and then we store audio transcript and detailed summaries of key frames in a PDF file.
5. Then we use Gemini to generate a very detailed summary.

## **Running the Script:**

1. First, make sure that you have everything to run this file in your system from requirement.txt.
2. Then replace your API key in the line:
    
    ```python
    genai.configure(api_key="YOUR_API_KEY")
    ```
    
3. Start the Streamlit app:
    
    ```python
    streamlit run app.py
    ```
    
4. Upload your video file through the Streamlit interface.


#### This project is done by Koleti Yashwanth Chowdary and Vishnupraneeswar part of Epoch Project Phase 2024-25
