# Import necessary libraries
import tkinter as tk
from tkinter import filedialog  # For opening file dialog to select images
from PIL import Image, ImageTk  # For handling image display in the GUI
import cv2  # OpenCV for real-time computer vision
import os  # For handling file paths
from threading import Thread  # For running tasks in the background
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

# Replace below paths withh corresponsing paths on system

# Paths for the model files and known faces directory
KNOWN_FACES_DIR = r"path/to/your/directory"  # Directory containing images of known faces

# Initialize the face analysis model
app = FaceAnalysis(r"path/to/your/model")
app.prepare(ctx_id=0, det_size=(640, 640))

# Global variables
cap = None  # Video capture object
frame = None  # Current frame from the video feed
known_faces = {}  # Dictionary to store embeddings of known faces
capture = False  # Flag to control video capture
recognition = False  # Flag to control face recognition
recognition_thread = None  # Thread for face recognition process

def load_known_faces(directory):
    """Load known faces from images in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust extensions as needed
            image_path = os.path.join(directory, filename)
            img = cv2.imread(image_path)
            
            # Extract faces and embeddings
            faces = app.get(img)
            for face in faces:
                embedding = face.embedding
                name = filename.split('.')[0]  # Use filename as the name (without extension)
                known_faces[name] = embedding
                print(f"Loaded face of {name} with embedding.")

# Load known faces from the specified directory
load_known_faces(KNOWN_FACES_DIR)

# GUI Setup
root = tk.Tk()
root.title("Real-Time Face Detection and Recognition System")
root.geometry("1000x600")

# Video capture frame
video_label = tk.Label(root)
video_label.place(x=10, y=10, width=640, height=480)

# Button States
video_capture_running = False
recognition_running = False

def process_video():
    """Handle the video processing loop."""
    while capture:
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB and display it in the GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            video_label.configure(image=img)
            video_label.image = img

        root.update()

def start_video_capture():
    """Start capturing video from the default camera and initialize video frame in the GUI."""
    global cap, capture, video_capture_running
    if not video_capture_running:
        cap = cv2.VideoCapture(0)  # Start capturing video from the default camera (usually webcam)
        capture = True
        video_capture_running = True
        process_video()

def stop_video_capture():
    """Stop video capture and clear the video frame in the GUI."""
    global capture, video_capture_running
    if video_capture_running:
        capture = False
        cap.release()
        video_label.configure(image='')  # Clear the video frame in the GUI
        video_capture_running = False

def start_recognition():
    """Start the face recognition process."""
    global recognition, recognition_thread
    if not recognition:
        recognition = True
        recognition_thread = Thread(target=recognize_faces_in_video)
        recognition_thread.daemon = True
        recognition_thread.start()

def stop_recognition():
    """Stop the face recognition process."""
    global recognition
    if recognition:
        recognition = False

def recognize_faces_in_video():
    """Process each frame from the video feed for face detection and recognition."""
    while capture and recognition:
        ret, frame = cap.read()
        if ret:
            # Detect faces in the frame
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)  # Bounding box coordinates
                embedding = face.embedding  # Face embedding (used for recognition)
                name = "Unknown"  # Default to unknown if no match is found
                color = (0, 0, 255)  # Red color for unknown faces

                # Compare detected face with known faces using cosine similarity
                highest_similarity = 0.5
                for known_name, known_embedding in known_faces.items():
                    similarity = cosine_similarity([embedding], [known_embedding])[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        name = known_name
                        color = (0, 255, 0)  # Green color for known faces

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert the frame to RGB and display it in the GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            video_label.configure(image=img)
            video_label.image = img

        root.update()

def feed_image():
    """Open a file dialog to select an image and perform face detection and recognition."""
    file_path = filedialog.askopenfilename()  # Open a file dialog to select an image
    img = cv2.imread(file_path)  # Read the selected image
    faces = app.get(img)
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        name = "Unknown"  # Default to unknown if no match is found
        color = (0, 0, 255)  # Red color for unknown faces

        # Compare with known faces using cosine similarity
        highest_similarity = 0.5
        for known_name, known_embedding in known_faces.items():
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                name = known_name
                color = (0, 255, 0)  # Green color for known faces

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Image Analysis", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GUI Buttons
start_capture_button = tk.Button(root, text="Start Capture", command=start_video_capture)
start_capture_button.place(x=660, y=10)

stop_capture_button = tk.Button(root, text="Stop Capture", command=stop_video_capture)
stop_capture_button.place(x=660, y=50)

start_recognition_button = tk.Button(root, text="Start Recognition", command=start_recognition)
start_recognition_button.place(x=660, y=90)

stop_recognition_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
stop_recognition_button.place(x=660, y=130)

feed_image_button = tk.Button(root, text="Feed Image", command=feed_image)
feed_image_button.place(x=660, y=170)

# Start the GUI event loop
root.mainloop()
