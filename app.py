import streamlit as st
import cv2
import numpy as np
import pytesseract
import pickle
from textblob import TextBlob

# Load trained models
with open('model_views.pkl', 'rb') as f:
    model_views = pickle.load(f)
with open('model_likes.pkl', 'rb') as f:
    model_likes = pickle.load(f)
with open('model_comments.pkl', 'rb') as f:
    model_comments = pickle.load(f)

st.title("📹 Video Upload & Performance Predictor")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
video_title = st.text_input("Video Title (for sentiment analysis)")

if uploaded_file is not None:
    temp_video = "temp_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(temp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * 3)
    
    frame_count = 0
    prev_gray = None
    motion_scores = []
    brightness_values = []
    has_face = False
    has_text = False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_scores.append(np.mean(diff))
        prev_gray = gray
        
        brightness_values.append(np.mean(gray))
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            has_face = True
        
        text = pytesseract.image_to_string(gray)
        if len(text.strip()) > 0:
            has_text = True
        
        frame_count += 1
    
    cap.release()
    
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    avg_brightness = np.mean(brightness_values) if brightness_values else 0
    
    st.subheader("🎥 Extracted Video Features")
    st.write(f"- Average Motion Score: {avg_motion:.4f}")
    st.write(f"- Average Brightness: {avg_brightness:.2f}")
    st.write(f"- Face Detected: {'Yes' if has_face else 'No'}")
    st.write(f"- Text Detected: {'Yes' if has_text else 'No'}")
    
    duration_input = st.number_input("Duration (Seconds)", min_value=1, max_value=60, value=3)
    
    if video_title:
        title_sentiment = TextBlob(video_title).sentiment.polarity
        st.write(f"Title Sentiment Score: {title_sentiment:.3f}")
        
        if st.button("Predict Performance"):
            X_input = np.array([[avg_motion, avg_brightness, int(has_face), int(has_text), duration_input, title_sentiment]])
            
            pred_views = model_views.predict(X_input)[0]
            pred_likes = model_likes.predict(X_input)[0]
            pred_comments = model_comments.predict(X_input)[0]
            
            st.subheader("📊 Predicted Performance")
            st.write(f"**Expected Views:** {int(pred_views):,}")
            st.write(f"**Expected Likes:** {int(pred_likes):,}")
            st.write(f"**Expected Comments:** {int(pred_comments):,}")
            
            tips = []
            if avg_brightness < 100:
                tips.append("Increase brightness; videos below 100 brightness often underperform.")
            if not has_text:
                tips.append("Add text overlays to engage viewers.")
            if duration_input < 10:
                tips.append("Short videos (<10s) may struggle; aim for at least 15s.")
            if title_sentiment < 0:
                tips.append("Consider a more positive title; negative sentiment may hurt engagement.")
            
            st.subheader("💡 Suggestions")
            if tips:
                for tip in tips:
                    st.write(f"- {tip}")
            else:
                st.write("Your video features look good compared to dataset averages!")
    else:
        st.warning("Please enter a video title for sentiment analysis.")
