import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
from gtts import gTTS
import io

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="SignSpeak",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Hardcoded Values ---
GEMINI_API_KEY = "AIzaSyC8DYC83s8v5CyIJm4Sq3lxZJefCKNeKSQ"
MODEL_PATH = "my_model.pt"

st.title("SignSpeak")
st.write("This app uses a finetuned model and API key. Select a source and click 'Start Processing' to begin.")

# ------------------ Sidebar for Inputs ------------------
st.sidebar.header("⚙️ Your Configuration")

st.sidebar.info(f"**Model:** `{MODEL_PATH}`")

# Confidence Threshold
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Source Selection
source_choice = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam"])

source_file = None
if source_choice == "Image":
    source_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'bmp'])
elif source_choice == "Video":
    source_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])

# ------------------ Main Logic ------------------

# Caching the model loading
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def configure_gemini(api_key):
    """Configures the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return None

def process_and_display(model, frame, confidence_thresh):
    """Processes a single frame for object detection and displays it."""
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    cropped_images_data = []
    
    bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
                   (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241)]

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        conf = detections[i].conf.item()
        classname = model.names[classidx]

        if conf > confidence_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_y - label_size[1] - 10),
                          (xmin + label_size[0], label_y + 5), color, -1)
            cv2.putText(frame, label, (xmin, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

            # Save cropped region in memory
            crop_img = frame[ymin:ymax, xmin:xmax]
            cropped_images_data.append(Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)))

    return frame, object_count, cropped_images_data

# --- Start Button ---
start_button = st.sidebar.button("🚀 Start Processing")

if start_button:
    # Check if model file exists before proceeding
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at the path '{MODEL_PATH}'. Make sure the model file is in the same directory as the app.")
        st.stop()

    gemini_model = configure_gemini(GEMINI_API_KEY)
    if not gemini_model:
        st.stop()

    yolo_model = load_yolo_model(MODEL_PATH)
    
    if not yolo_model:
        st.stop()

    # --- Processing based on source ---
    if source_choice == "Image" and source_file:
        image = Image.open(source_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed_frame, obj_count, cropped_images = process_and_display(yolo_model, frame, confidence_thresh)
        
        st.subheader(f"Processed Image (Detected {obj_count} objects)")
        st.image(processed_frame, channels="BGR", width='stretch')

        # --- OCR and TTS Section ---
        if cropped_images:
            st.subheader("Text Recognition (OCR)")
            all_text = ""
            with st.spinner("Gemini is reading the detected objects..."):
                for i, img in enumerate(cropped_images):
                    try:
                        prompt = "What text do you see in this image? Only return the visible text, no explanation."
                        response = gemini_model.generate_content([prompt, img])
                        text = response.text.strip()
                        if text.lower() != "no text found" and text:
                            st.info(f"**Object {i+1}:** `{text}`")
                            all_text += text + ". "
                    except Exception as e:
                        st.error(f"Error during OCR for object {i+1}: {e}")
            
            if all_text.strip():
                st.subheader("Text-to-Speech")
                with st.spinner("Generating audio..."):
                    try:
                        tts = gTTS(all_text)
                        mp3_fp = io.BytesIO()
                        tts.write_to_fp(mp3_fp)
                        st.audio(mp3_fp, format='audio/mp3')
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
            else:
               st.warning("No text was detected in any of the objects.")


    elif source_choice in ["Video", "Webcam"]:
        if source_choice == "Video" and not source_file:
            st.warning("Please upload a video file.")
            st.stop()

        cap = None
        if source_choice == "Webcam":
            cap = cv2.VideoCapture(0)
        else: # Video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(source_file.read())
                video_path = tfile.name
                cap = cv2.VideoCapture(video_path)

        st.subheader("Video Stream")
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        stop_button = st.button("⏹️ Stop Processing")
        
        frame_count = 0
        OCR_INTERVAL = 60
        last_spoken_text = ""
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.write("Video stream ended.")
                break

            start_time = time.perf_counter()
            processed_frame, obj_count, cropped_images = process_and_display(yolo_model, frame, confidence_thresh)
            
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            info_placeholder.info(f"Objects Detected: {obj_count} | FPS: {fps:.2f}")
            frame_placeholder.image(processed_frame, channels="BGR", width='stretch')
            
            # --- Controlled OCR + TTS ---
            frame_count += 1
            if frame_count % OCR_INTERVAL == 0 and cropped_images:
                st.subheader("Text Recognition (OCR)")
                all_text = ""
                with st.spinner(f"Gemini is reading frame {frame_count}..."):
                    for i, img in enumerate(cropped_images):
                         try:
                             prompt = "What text do you see in this image? Only return the visible text, no explanation."
                             response = gemini_model.generate_content([prompt, img])
                             text = response.text.strip()
                             if text.lower() != "no text found" and text:
                                 st.info(f"**Object {i+1}:** `{text}`")
                                 all_text += text + ". "
                         except Exception as e:
                             st.error(f"Error during OCR for object {i+1}: {e}")
                
                # --- MODIFIED LOGIC ---
                # Only generate audio if the new text is different from the last one
                current_text = all_text.strip()
                if current_text and current_text != last_spoken_text:
                    st.subheader("Text-to-Speech")
                    with st.spinner("Generating new audio..."):
                        try:
                            tts = gTTS(current_text)
                            mp3_fp = io.BytesIO()
                            tts.write_to_fp(mp3_fp)
                            st.audio(mp3_fp, format='audio/mp3')
                            last_spoken_text = current_text # Update the last spoken text
                        except Exception as e:
                            st.error(f"TTS failed: {e}")
                elif not current_text:
                    st.warning(f"No text detected in frame {frame_count}.")


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

