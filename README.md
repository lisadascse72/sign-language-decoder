# 🤖 Sign Language Decoder

A **Streamlit-based web app** that uses **YOLOv8** for recognizing letters from **sign language gestures**. Users can upload images or videos, or use their webcam to detect hand signs in real-time. The app also provides audio feedback for detected letters.

> ⚡ This project is a customized version of the original work by [Mariam Gamal](https://github.com/Mariam111) with additional modifications and simplifications.

---

## 🎯 Features

- ✅ Detect **sign language letters** from:
  - Uploaded images
  - Uploaded or sample videos
  - Webcam in real-time
- ✅ Display of **detected letters** on-screen
- ✅ **Text-to-speech** for detected letters (using `pyttsx3`)
- ✅ YOLOv8 model-based accurate predictions
- ✅ Clean and minimal interface using **Streamlit**

---

## 🔧 Tech Stack

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lisadascse72/sign-language-decoder.git
cd sign-language-decoder


```bash
  git clone https://github.com/Mariam111/Sign-Language-Recognition
```

Go to the project directory

```bash
  cd Sign-Language-Recognition
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python -m streamlit run app.py
```

#original Author

- [Mariam Gamal](https://github.com/Mariam111)


