import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import faiss
import pickle
import time
import pyttsx3
import threading

st.set_page_config(page_title="ASL to Text & Speech", layout="centered")

st.title("American Sign Language to Text and Speech Converter")

# ---------------- LOAD FAISS ----------------
INDEX_PATH = "index_labels/faiss_index_ivfflat.index"
LABEL_PATH = "index_labels/index_to_label.pkl"

@st.cache_resource
def load_faiss():
    index = faiss.read_index(INDEX_PATH)
    with open(LABEL_PATH, "rb") as f:
        labels = pickle.load(f)
    return index, labels

index, labels = load_faiss()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- SESSION STATE ----------------
if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "last_label" not in st.session_state:
    st.session_state.last_label = None
if "label_start_time" not in st.session_state:
    st.session_state.label_start_time = 0.0
if "cap" not in st.session_state:
    st.session_state.cap = None

hold_time = 1.0

# ---------------- CHECKBOX CONTROLS ----------------
start_camera = st.checkbox("Start Camera")

if start_camera:
    start_recognition = st.checkbox("Start Recognition")
else:
    start_recognition = False

# ---------------- PLACEHOLDERS ----------------
frame_placeholder = st.image([])
sentence_placeholder = st.empty()

# ---------------- NON-BLOCKING SPEECH ----------------
def speak_async(text):
    def _speak():
        engine = pyttsx3.init(driverName="sapi5")
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_speak, daemon=True).start()

# ---------------- ACTION BUTTONS ----------------
if start_camera:
    col1, col2 = st.columns(2)

    with col1:
        speak_clicked = st.button("Speak Full Sentence")
    with col2:
        clear_clicked = st.button("Clear Sentence")
else:
    speak_clicked = False
    clear_clicked = False

if speak_clicked and st.session_state.sentence.strip():
    speak_async(st.session_state.sentence)

if clear_clicked:
    st.session_state.sentence = ""

# ---------------- CAMERA LOOP ----------------
if start_camera:

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap

    while start_camera:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if start_recognition and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                embedding = np.array(coords, dtype=np.float32).reshape(1, -1)
                embedding /= np.linalg.norm(embedding)

                _, I = index.search(embedding, 1)
                recognized_label = labels[I[0][0]]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                now = time.time()

                # -------- EXACT sign_recognition.py LOGIC --------
                if recognized_label == st.session_state.last_label:
                    if now - st.session_state.label_start_time >= hold_time:

                        if recognized_label == "space":
                            st.session_state.sentence += " "
                        elif recognized_label == "del":
                            st.session_state.sentence = st.session_state.sentence[:-1]
                        elif recognized_label == "nothing":
                            pass
                        elif recognized_label.isalpha() and len(recognized_label) == 1:
                            st.session_state.sentence += recognized_label

                        st.session_state.last_label = None
                else:
                    st.session_state.last_label = recognized_label
                    st.session_state.label_start_time = now

                cv2.putText(
                    frame,
                    f"Detected: {recognized_label.upper()}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

        status = "DETECTING" if start_recognition else "IDLE"
        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(
            frame,
            st.session_state.sentence,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        sentence_placeholder.markdown(
            f"### Current Sentence\n<span style='color:white;font-size:22px'>{st.session_state.sentence}</span>",
            unsafe_allow_html=True
        )

        time.sleep(0.03)

else:
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None