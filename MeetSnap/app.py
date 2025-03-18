import cv2
import os
import re
import numpy as np
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import multiprocessing
import subprocess
import threading

import google.generativeai as genai
import base64
import os
import requests
from reportlab.lib.units import inch
import time
import google.api_core.exceptions

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Keep api keys from Google AI Studio in next line
genai.configure(api_key="")
transcript = None

keyf = []


def naturally_sorted_filenames(dir_path):
    try:
        if not os.path.isdir(dir_path):
            raise ValueError(f"The provided path '{dir_path}' is not a valid directory.")
        filenames = [filename for filename in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, filename))]
        return sorted(filenames, key=lambda filename: [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', filename)])

    except Exception as e:
        print(f"Error: {e}")
        return []

import subprocess

def extract_audio(video_path, output_audio_path):
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")
    global transcript
    command = [
        "ffmpeg",
        "-hwaccel", "cuda", 
        "-i", video_path,      
        "-q:a", "10",           
        "-map", "a",           
        output_audio_path     
    ]

    # Run FFmpeg command
    subprocess.run(command, check=True)
    print(f"Audio extracted and saved to {output_audio_path}")
    myfile = genai.upload_file("audio.mp3")
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    result = model.generate_content([myfile, "get transcript from the  audio"])
    transcript = result.text


def cal_diff(image_path1, image_path2):
    # Load images to GPU
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    # Upload to GPU
    gpu_img1 = cv2.cuda_GpuMat()
    gpu_img1.upload(img1)
    gpu_img2 = cv2.cuda_GpuMat()
    gpu_img2.upload(img2)
    
    # GPU-accelerated absolute difference
    gpu_diff = cv2.cuda.absdiff(gpu_img1, gpu_img2)
    cpu_diff = gpu_diff.download()
    
    l2_norm = np.linalg.norm(cpu_diff)
    return l2_norm


def capture_frames(path):
    output_folder = "video"
    os.makedirs(output_folder, exist_ok=True)

    # Using FFmpeg with GPU acceleration for frame extraction
    command = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-i", path,
        "-vf", "fps=1",  # 1 frame per second
        os.path.join(output_folder, "%d.jpg")
    ]
    subprocess.run(command, check=True)

    # Step 2: Compute dynamic threshold
    image_files = sorted(os.listdir(output_folder), key=lambda x: int(os.path.splitext(x)[0]))
    if len(image_files) < 2:
        print("Not enough images to compute differences.")
        return

    # Calculate differences between consecutive frames
    diffs = []
    for idx in range(len(image_files) - 1):
        img1 = os.path.join(output_folder, image_files[idx])
        img2 = os.path.join(output_folder, image_files[idx + 1])
        diffs.append(cal_diff(img1, img2))

    # Compute mean and standard deviation of differences
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    dynamic_threshold = mean_diff + 2 * std_diff
    print(f"Dynamic threshold: {dynamic_threshold}")

    # Step 3: Select key frames using dynamic threshold
    key_frames_folder = "key_frames"
    os.makedirs(key_frames_folder, exist_ok=True)
    j = 1
    i = 0

    while i < len(image_files) - 1:
        img1 = os.path.join(output_folder, image_files[i])
        img2 = os.path.join(output_folder, image_files[i + 1])
        l2_norm = cal_diff(img1, img2)

        if l2_norm > max(min(15000, dynamic_threshold), 10000):
            k = i + 1
            while k + 1 < len(image_files) and cal_diff(
                os.path.join(output_folder, image_files[k]),
                os.path.join(output_folder, image_files[k + 1])
            ) < dynamic_threshold:
                k += 1

            # Save the key frame
            key_frame = cv2.imread(os.path.join(output_folder, image_files[k]))
            cv2.imwrite(os.path.join(key_frames_folder, f"{j}.jpg"), key_frame)
            j += 1
            i = k
        else:
            i += 1

    important_frames()  # Now this will execute

    return


def important_frames():
  image_files = naturally_sorted_filenames("key_frames")
  summary = []
  model = genai.GenerativeModel(model_name = "gemini-1.5-flash-8b")
  image_directory = "key_frames"

  for i, filename in enumerate(image_files, start=1):
    #print(i)
    image_path = os.path.join(image_directory, filename)

    with open(image_path, "rb") as image_file:
        image = image_file.read()

    prompt = "caption this image, dont give any extra output like Here's a caption summarizing the image"

    try:
        response = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': base64.b64encode(image).decode('utf-8')}, prompt]
        )
        #print(f"Caption for {filename}: {response.text}")
    except requests.exceptions.RequestException as e:
      if response is not None and response.status_code == 429:
        print(f"Request failed with 429 (Too Many Requests): {e}")
        print(f"Rate limit exceeded for {filename}. Waiting 20 seconds...")
        time.sleep(10)
        response = model.generate_content(
            [{'mime_type': 'image/jpeg', 'data': base64.b64encode(image).decode('utf-8')}, prompt]
        )
        #print(f"Caption for {filename}: {response.text}")

    summary.append(response.text)
    #time.sleep(5)

  hey = summary.copy()
  for i in range(len(hey)):
    hey[i] = hey[i].replace("\n","")
    hey[i] = str(i+1)+". "+hey[i]

  prompt = '''
    You are a video summarizer. Your input is a sequence of image captions representing video frames. Your task is to analyze the sequence and identify unnecessary frames based on the following criteria:

    Criteria for Unnecessary Frames:
    Redundant Frames:

    Consecutive or non-consecutive frames that are visually or contextually repetitive.
    Scenes with minimal or no meaningful change.
    Slight variations (e.g., minor object movement, blinking, subtle lighting changes) that do not add value.
    Images which says only about person in the image
    Irrelevant Frames:

    Frames that do not contribute meaningfully to the video's storyline, context, or purpose.
    Credits, intros, outros, watermarks, logos, disclaimers, or promotional elements.
    Frames containing static text or unrelated graphics that do not advance the core content.
    Filler frames with blank screens, excessive transitions, or background elements with no action.
    Low-Information Frames:

    Completely blurred, overexposed, underexposed, or visually unclear frames.
    Frames with large empty spaces or static backgrounds without meaningful action.
    Output Format:
    Provide only the indices of unnecessary frames, separated by commas or spaces if multiple. Do not include any additional text, labels, context, or explanation—output only the numbers.
    '''
  for i in range(len(hey)):
     prompt = prompt + hey[i] + "\n"

  model = genai.GenerativeModel(model_name = "gemini-1.5-flash")
  prompt_list = [{'mime_type': 'text/plain', 'data': prompt.encode('utf-8') }]
  response = model.generate_content(prompt_list,
  generation_config=genai.types.GenerationConfig(temperature=0.0))

  all_frames_folder = "key_frames"
  all_frame_files = [f for f in os.listdir(all_frames_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
  all_frame_numbers = [int(re.search(r'\d+', f).group()) for f in all_frame_files]

  numbers = re.findall(r'\d+', response.text)
  numbers = [int(num) for num in numbers]
  distinct_numbers = list(set(numbers))
  print(distinct_numbers)
  frames_to_copy = [frame for frame in all_frame_numbers if frame not in distinct_numbers]
  new_folder = "selected_frames"
  os.makedirs(new_folder, exist_ok=True)
  frames_to_copy.sort()
  for i, frame_number in enumerate(frames_to_copy):
      frame_filename = f"{frame_number}.jpg"
      new_filename = f"{i + 1}.jpg"
      source_path = os.path.join(all_frames_folder, frame_filename)
      destination_path = os.path.join(new_folder, new_filename)
      shutil.copy(source_path, destination_path)


  return

def process_single_image(filename, max_retries=1, delay=5):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    image_path = os.path.join("selected_frames", filename)

    with open(image_path, "rb") as image_file:
        image = image_file.read()

    prompt = "give a very detailed explanation of the image"

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [{'mime_type': 'image/jpeg', 'data': base64.b64encode(image).decode('utf-8')}, prompt]
            )
            return response.text  

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed after {max_retries} attempts. Skipping {filename}.")
    
    return None 

def summary_images():
    print("Processing all images in parallel with retry handling")
    global keyf
    image_files = naturally_sorted_filenames("selected_frames")
    
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers based on API limits
        results = list(executor.map(process_single_image, image_files))
    
    global keyf
    keyf = [result for result in results if result]
    
    for i, result in enumerate(keyf, start=1):
        print(f"{i}. {result}")
    
    print("All images processed successfully!")

    return

def process_pdf(pdf_path, prompt, temperature=1, max_output_tokens=8192):
    model = genai.GenerativeModel("gemini-1.5-flash") 
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = base64.standard_b64encode(pdf_file.read()).decode("utf-8")

    response = model.generate_content(
        contents=[
            {'mime_type': 'application/pdf', 'data': pdf_data},
            prompt
        ],
        generation_config=genai.types.GenerationConfig( 
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    )
    return response.text 

def make_pdf():
    global keyf
    hey = keyf.copy()
    for i in range(len(hey)):
        hey[i] = str(i+1)+". "+keyf[i]

    doc = SimpleDocTemplate("summary.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add audio_to_text
    story.append(Paragraph("Audio Transcription:", styles['h1']))
    story.append(Paragraph(transcript, styles['Normal']))
    story.append(Spacer(1, 12)) 

    # Add keyf
    story.append(Paragraph("Key Frame Explanations:", styles['h1']))
    
    for item in hey:
        story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))  

    doc.build(story)
    prompt = "You are a video summarizer tasked with analyzing a document or transcript derived from a video, including any associated slide text or visuals. Your job is to create a detailed, well-structured summary by performing the following:Subsection Identification:Carefully analyze the content to determine logical subsections based on the topics discussed in the video.Create clear and descriptive titles for each subsection to reflect the core ideas or themes.Highlight Extraction:Identify the most important points within each subsection, including:Key concepts, definitions, and principles.Critical data, statistics, or facts mentioned in the video.Mathematical equations (if any) with detailed explanations of their components and relevance.Practical examples, case studies, or real-world applications.Visuals (e.g., graphs, charts, or images) and their significance to the topic.Content Organization:Present the summary in a logical, easy-to-follow format that mirrors the flow of the video or enhances it if the original lacks clear structure.Ensure a coherent narrative by connecting related points within and across subsections.Focus on Relevance:Highlight only the essential details and avoid including unrelated or redundant information.Exclude credits, references, or any irrelevant aspects of the video.Output Format:Use a hierarchical structure with:Section titles and subtitles.Bullet points or numbered lists for clarity.Explanatory notes for complex topics or visuals.Autonomous Structuring:If the content lacks explicit organization, infer the structure based on the main ideas and flow of the video.Context and Purpose:Incorporate the broader purpose or context of the video, emphasizing its key messages, themes, or takeaways.The final summary should be clear, detailed, and well-organized, offering a comprehensive understanding of the video’s content while excluding irrelevant details or credits."
    pdf_file_path = "summary.pdf" 
    your_prompt = prompt

    summary = process_pdf(pdf_file_path, your_prompt, temperature=0.5, max_output_tokens=8192) # Example values
    print(summary)

    st.header('Summary')
    st.write(summary)

    folder_path = "selected_frames"
    st.header('Key Frames')

    images_files = naturally_sorted_filenames(folder_path)

    # Display images
    for image_file in images_files:
        image_path = os.path.join(folder_path, image_file)
        st.image(image_path)




def main():
    st.title("Video Summarizer App")


    folder_path = ["selected_frames","key_frames","video"]  
    for folder in folder_path:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            continue



    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success("Video uploaded successfully!")

        start = time.time()

        process1 = multiprocessing.Process(target=capture_frames, args=(video_path,))

        thread2 = threading.Thread(target=extract_audio, args=(video_path, "audio.mp3"))

        process1.start()
        thread2.start()

        process1.join()
        thread2.join()
        print("Both tasks completed!")

        summary_images()

        make_pdf()

        end = time.time()
        print("Total execution time:", end - start)


if __name__ == "__main__":
    main()
