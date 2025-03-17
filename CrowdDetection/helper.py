from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import tempfile
from ultralytics.solutions import object_counter
from pathlib import Path
import settings
from collections import defaultdict






FILE = Path(__file__).resolve()

ROOT = FILE.parent
ROOT = ROOT.relative_to(Path.cwd())
MODEL_DIR = ROOT / 'weights'
VIDEOS_DIR = ROOT / 'videos'
opPath = VIDEOS_DIR / 'OP'
modelPath=MODEL_DIR / 'yolo11n.pt'



def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None



def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display detected objects on a video frame using the YOLO model and count objects.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YOLO): A YOLO object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): Whether to display object tracking.

    Returns:
    None
    """
    # Resize the image for consistency
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Perform object detection
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
        #res = model.predict(image, conf=conf, iou=0.8)  # Disable tracking
    else:
        res = model.predict(image, conf=conf, iou=0.8)

    detected_boxes = res[0].boxes
    detected_classes = res[0].boxes.cls.cpu().numpy()


    from collections import Counter
    class_counts = Counter(detected_classes)  # Example: {0: 6, 32: 2} (6 persons, 2 rackets)

    CLASS_NAMES = model.names  

    detection_summary = ", ".join([f"{int(count)} {CLASS_NAMES[int(cls)]}" for cls, count in class_counts.items()])
    print(f"DEBUG: YOLO detected -> {detection_summary}")  # Print exact YOLO counts

    
    
    print("DEBUG: Processing new frame...")
    #res = model.predict(image, conf=0.5, iou=0.8)  # Force fresh detections
    
    #detected_boxes = res[0].boxes
    print(f"DEBUG: Detected {len(detected_boxes)} objects in this frame.")
    
    if len(detected_boxes) > 0:
        for i, box in enumerate(detected_boxes):
            print(f"DEBUGG: Object {i+1}: {box.xyxy.cpu().numpy()}")
    else:
        print("DEBUG: No objects detected in this frame.")


    

    # Initialize the object counter
    counter = object_counter.ObjectCounter(view_img=True, draw_tracks=True)

    '''processed_frame, count_result = counter.count(image, return_counts=True)
    print(f"DEBUG: ObjectCounter counts {count_result}")'''

    # Process the frame and count objects
    processed_frame = counter.count(image)

    # Count objects manually
    total_objects = len(detected_boxes)
    #print("LAALALALALALA",detected_boxes)

    # Debug: Print all detected boxes
    '''for i, box in enumerate(detected_boxes):
        print(f"DEBUGG: Object {i+1}: {box.xyxy.cpu().numpy()}") # Print bounding box coordinates'''

    # Overlay total count on the frame
    '''cv2.putText(processed_frame, f"Objects: {total_objects}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)'''

    cv2.putText(processed_frame, detection_summary, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with object count in Streamlit
    #st_frame.image(processed_frame, caption='Detected Video', channels="BGR", use_column_width=True)
    st_frame.image(processed_frame, caption='Detected Video', channels="BGR", use_container_width=True)


    '''print('In Counts : {}'.format(counter.in_counts))     # display in counts
    print("Out Counts : {}".format(counter.out_counts))   # display out counts'''
'''
def _display_detected_frames(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    #video_writer.release()
    #cv2.destroyAllWindows()'''
    
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'no_warnings': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(
                        conf,
                        model,
                        st_frame,
                        image,
                        is_display_tracker,
                        tracker
                    )
                else:
                    break

            vid_cap.release()

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model, model_path):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    '''source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())'''

    
    #source_vid = st.sidebar.file_uploader(
        #"Choose a video...")

    source_vid = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    

    is_display_tracker, tracker = display_tracker_options()

    if source_vid is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(source_vid.read())
            temp_video_path = temp_video.name

    '''
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
                    '''

    #with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
    '''with open(source_vid, 'rb') as video_file:
        video_bytes = video_file.read()'''
    video_bytes = source_vid
    if video_bytes:
        st.video(video_bytes)

    if source_vid is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(source_vid.read())
            temp_video_path = temp_video.name

    if st.sidebar.button('Detect Video Objects') and source_vid is not None:
        try:
            vid_cap = cv2.VideoCapture(
                #str(settings.VIDEOS_DICT.get(source_vid)))
                str(temp_video_path))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                

                success, image = vid_cap.read()
                if success:
                    
                    
                    

                    '''
                    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
                    w, h, fps = (int(vid_cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
                    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)
                    while vid_cap.isOpened():
                        success, im0 = vid_cap.read()
                        if not success:
                            print("Video frame is empty or video processing has been successfully completed.")
                            break
                        im0 = counter.count(im0)
                        video_writer = cv2.VideoWriter(ROOT / 'videos', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        video_writer.write(im0)'''

                    #_display_detected_frames(str(temp_video_path),opPath,modelPath)
                    
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
                
            
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
