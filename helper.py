from ultralytics import YOLO
import streamlit as st
import cv2
import settings
from gemini_helper import predict_sentence_from_letters  # 🔮 Gemini API

# 🔡 Class ID to Alphabet Mapping
CLASS_NAMES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

BUFFER_LIMIT = 10

def load_model(model_path):
    return YOLO(model_path)

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'), key="tracker_toggle")
    is_display_tracker = display_tracker == 'Yes'
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"), key="tracker_choice")
        return is_display_tracker, tracker_type
    return False, None

def get_detected_letter(results):
    detections = results[0].boxes.data.cpu().numpy()
    if len(detections) > 0:
        class_id = int(detections[0][5])
        return CLASS_NAMES.get(class_id, '?')
    return None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    if "recent_letters" not in st.session_state:
        st.session_state["recent_letters"] = []

    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    if is_display_tracking:
        results = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        results = model.predict(image, conf=conf)

    res_plotted = results[0].plot()
    st_frame.image(res_plotted, caption='Detected Frame', channels="BGR", use_container_width=True)

    letter = get_detected_letter(results)
    if letter:
        if not st.session_state["recent_letters"] or st.session_state["recent_letters"][-1] != letter:
            st.session_state["recent_letters"].append(letter)
            if len(st.session_state["recent_letters"]) > BUFFER_LIMIT:
                st.session_state["recent_letters"].pop(0)

        st.success(f"Detected Letter: {letter}")
        st.markdown(f"🅰️ **Recent Letters**: `{''.join(st.session_state['recent_letters'])}`")

        if st.session_state.get("show_sentence", True):
            predicted = predict_sentence_from_letters("".join(st.session_state["recent_letters"]))
            st.info(f"🧠 **Predicted Word/Sentence**: `{predicted}`")

def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('▶️ Detect Webcam Signs', key="webcam_button"):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                image = cv2.flip(image, 1)
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error accessing webcam.")
            st.exception(e)

def play_stored_video(conf, model):
    is_upload = st.sidebar.radio("🎦 Video Source", ["Sample Videos", "Upload Your Own"], key="video_source_toggle")
    is_display_tracker, tracker = display_tracker_options()

    video_path = None
    if is_upload == "Sample Videos":
        source_vid = st.sidebar.selectbox("🎞️ Choose a video...", settings.VIDEOS_DICT.keys(), key="video_selector")
        video_path = settings.VIDEOS_DICT.get(source_vid)
    else:
        uploaded_video = st.sidebar.file_uploader("📤 Upload your video", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video:
            video_path = "temp_uploaded_video.mp4"
            with open(video_path, "wb") as out_file:
                out_file.write(uploaded_video.read())
        else:
            st.warning("Please upload a video.")
            return

    if video_path:
        st.video(video_path)

    if st.sidebar.button('▶️ Detect Video Signs', key="video_button"):
        try:
            vid_cap = cv2.VideoCapture(str(video_path))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error processing video.")
            st.exception(e)
