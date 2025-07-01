from pathlib import Path
import PIL
import pyttsx3
import cv2
import streamlit as st
import settings
import helper

# ✅ Set Streamlit page config
st.set_page_config(
    page_title="Sign Language Recognition using YOLOv8",
    page_icon="🧏‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Page title
st.title("🤖 Sign Language Recognition using YOLOv8")

# ✅ Sidebar - Confidence slider
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# ✅ Load model
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"⚠️ Unable to load model. Check the path: `{model_path}`")
    st.exception(ex)
    st.stop()

# ✅ Text-to-speech for detected letter
def speak_detected_letter():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        letter = st.session_state.get("last_detected_letter", None)
        if letter:
            engine.say(f"The detected letter is {letter}")
            engine.runAndWait()
    except Exception as e:
        st.error("Speech engine failed.")
        st.exception(e)

# ✅ Sidebar - Source Configuration
st.sidebar.header("Source Configuration")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# ✅ IMAGE MODE
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(default_image_path, caption="Default Image", use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        except Exception as ex:
            st.error("⚠️ Error while opening the image.")
            st.exception(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected_image_path, caption='Detected Output', use_container_width=True)
        else:
            if st.sidebar.button('🔍 Detect Signs in Image', key="detect_image"):
                try:
                    results = model.predict(uploaded_image, conf=confidence)
                    boxes = results[0].boxes
                    detected_img = results[0].plot()[:, :, ::-1]
                    st.image(detected_img, caption='Detected Image', use_container_width=True)

                    detections = boxes.data.cpu().numpy()
                    if len(detections) > 0:
                        class_id = int(detections[0][5])
                        letter = helper.CLASS_NAMES.get(class_id, '?')
                        st.success(f"Detected Letter: {letter}")
                        st.session_state["last_detected_letter"] = letter

                    with st.expander("📋 Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.error("⚠️ Detection failed.")
                    st.exception(ex)

            if "last_detected_letter" in st.session_state:
                if st.button("🔊 Speak Detected Letter", key="speak_image"):
                    speak_detected_letter()

# ✅ VIDEO MODE
elif source_radio == settings.VIDEO:
    uploaded_video = st.sidebar.file_uploader("Upload your video...", type=["mp4", "avi", "mov", "mkv"], key="video_upload")

    if uploaded_video:
        with open("temp_uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.video("temp_uploaded_video.mp4")

        if st.sidebar.button("▶️ Detect Signs in Uploaded Video", key="detect_uploaded_video"):
            try:
                vid_cap = cv2.VideoCapture("temp_uploaded_video.mp4")
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        helper._display_detected_frames(confidence, model, st_frame, image)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error processing uploaded video.")
                st.exception(e)
    else:
        helper.play_stored_video(confidence, model)

# ✅ WEBCAM MODE
elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

# ❌ Fallback
else:
    st.error("Please select a valid source from the sidebar!")
