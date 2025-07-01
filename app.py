import streamlit as st
import numpy as np
import pytesseract
import pickle
from textblob import TextBlob
from PIL import Image
from moviepy.editor import VideoFileClip

# Load trained models
with open('model_views.pkl', 'rb') as f:
    model_views = pickle.load(f)
with open('model_likes.pkl', 'rb') as f:
    model_likes = pickle.load(f)
with open('model_comments.pkl', 'rb') as f:
    model_comments = pickle.load(f)

st.title("📹 Video Upload & Performance Predictor (No OpenCV)")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
video_title = st.text_input("Video Title (for sentiment analysis)")

if uploaded_file is not None:
    temp_video = "temp_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())

    clip = VideoFileClip(temp_video)
    fps = clip.fps
    duration = min(3, clip.duration)  # use up to first 3 seconds
    total_frames = int(fps * duration)

    st.write(f"Processing first {duration:.1f} seconds at {fps:.1f} fps ({total_frames} frames)...")

    frame_count = 0
    prev_gray = None
    motion_scores = []
    brightness_values = []
    has_text = False

    for frame in clip.iter_frames(fps=fps, dtype="uint8"):
        if frame_count >= total_frames:
            break

        gray = np.mean(frame, axis=2).astype(np.uint8)  # convert RGB → grayscale

        if prev_gray is not None:
            diff = np.abs(prev_gray.astype(np.int16) - gray.astype(np.int16))
            motion_scores.append(np.mean(diff))
        prev_gray = gray

        brightness_values.append(np.mean(gray))

        pil_img = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_img)
        if text.strip():
            has_text = True

        frame_count += 1

    avg_motion = np.mean(motion_scores) if motion_scores else 0
    avg_brightness = np.mean(brightness_values) if brightness_values else 0

    st.subheader("🎥 Extracted Video Features")
    st.write(f"- Average Motion Score: {avg_motion:.4f}")
    st.write(f"- Average Brightness: {avg_brightness:.2f}")
    st.write(f"- Text Detected: {'Yes' if has_text else 'No'}")

    duration_input = st.number_input("Duration (Seconds)", min_value=1, max_value=60, value=int(duration))

    if video_title:
        title_sentiment = TextBlob(video_title).sentiment.polarity
        st.write(f"Title Sentiment Score: {title_sentiment:.3f}")

        if st.button("Predict Performance"):
            X_input = np.array([[avg_motion, avg_brightness, 0, int(has_text), duration_input, title_sentiment]])
            # Note: Face detection removed → always 0

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
