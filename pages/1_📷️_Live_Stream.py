import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import importlib.util  # Import for dynamic module loading

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

st.title('FormMaster')

# Dropdown for selecting exercises
exercise_options = ['Squats', 'Bicep Curls', 'Pushups', 'Shoulder Press', 'Tricep Extensions']
selected_exercise = st.selectbox('Select Exercise', exercise_options)

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

thresholds = None 

if mode == 'Beginner':
    thresholds = get_thresholds_beginner()
elif mode == 'Pro':
    thresholds = get_thresholds_pro()

# Initialize the ProcessFrame instance based on the selected exercise
if selected_exercise == 'Shoulder Press':
    # Dynamically import and run the shoulder_press.py code
    shoulder_press_path = "D:\\main project\\FormMaster\\shoulder_press.py"
    spec = importlib.util.spec_from_file_location("shoulder_press", shoulder_press_path)
    shoulder_press_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shoulder_press_module)
    
    # Display comments or instructions for Shoulder Press
    st.info("Perform the Shoulder Press by lifting weights overhead. Maintain good posture and control.")
    
    # Initialize the ProcessFrame instance for Shoulder Press
    live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
else:
    live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)  # For other exercises

# Initialize face mesh solution
pose = get_mediapipe_pose()

if 'download' not in st.session_state:
    st.session_state['download'] = False

output_video_file = f'output_live.flv'

def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
    frame, _ = live_process_frame.process(frame, pose)  # Process frame
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # Encode and return BGR frame

def out_recorder_factory() -> MediaRecorder:
    return MediaRecorder(output_video_file)

ctx = webrtc_streamer(
    key="Squats-pose-analysis",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
    media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    out_recorder_factory=out_recorder_factory
)

download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data=op_vid, file_name='output_live.flv')

        if download:
            st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()