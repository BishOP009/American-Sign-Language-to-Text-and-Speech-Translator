import cv2
import mediapipe as mp
import numpy as np
import faiss
import pickle
import time
import pyttsx3   # <-- Added for Text-to-Speech

# Load FAISS index and labels
index = faiss.read_index("F:/Sign Language Converter/index_labels/faiss_index_ivfflat.index")
with open("F:/Sign Language Converter/index_labels/index_to_label.pkl", "rb") as f:
    labels = pickle.load(f)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting real-time hand sign recognition... Press 's' to toggle detection, 'r' to read full sentence, 'q' to quit.")

sentence = ""             
last_label = None         
label_start_time = 0      

hold_time = 1    # reduced from 1.5s → faster recognition
detecting = False  # start in idle mode

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if detecting and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract normalized landmark coordinates
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            embedding = np.array(coords, dtype=np.float32).reshape(1, -1)

            # Normalize embedding (L2 norm)
            embedding /= np.linalg.norm(embedding)

            # Search in FAISS
            D, I = index.search(embedding, 1)
            recognized_label = labels[I[0][0]]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if label is stable for enough time
            current_time = time.time()
            if recognized_label == last_label:
                if current_time - label_start_time >= hold_time:
                    if recognized_label == "space":
                        sentence += " "
                        words = sentence.strip().split()
                        if words:  # If there is at least one word
                            last_word = words[-1]
                            print(f"🔊 Speaking word: {last_word}")
                            # Re-initialize engine for each word
                            word_engine = pyttsx3.init(driverName='sapi5')
                            word_engine.setProperty('rate', 150)
                            word_engine.setProperty('volume', 1.0)
                            word_engine.say(last_word)
                            word_engine.runAndWait()
                            del word_engine
                        print("✔ Added: [space]")

                    elif recognized_label == "del":
                        if sentence:
                            sentence = sentence[:-1]
                            print("✔ Deleted last character")

                    elif recognized_label == "nothing":
                        # Just ignore "nothing"
                        pass

                    elif recognized_label.isalpha() and len(recognized_label) == 1:
                        sentence += recognized_label
                        print(f"✔ Added: {recognized_label}")

                    print(f"Current Sentence: {sentence}")
                    last_label = None
            else:
                last_label = recognized_label
                label_start_time = current_time

    # Show current sentence on screen
    cv2.putText(frame, sentence, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Detection hint overlay
    if detecting:
        cv2.putText(frame, f"Hold steady for {hold_time:.1f}s", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Idle Mode (Press 's' to start)", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture to Sentence", frame)

    # Handle key inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        detecting = not detecting
        print("🟢 Detection ON" if detecting else "🔴 Detection OFF")

    elif key == ord("r"):  # <-- Read full sentence
        if sentence.strip():
            print(f"🔊 Reading full sentence: {sentence}")
            speak_engine = pyttsx3.init(driverName='sapi5')
            speak_engine.setProperty('rate', 150)
            speak_engine.setProperty('volume', 1.0)
            speak_engine.say(sentence)
            speak_engine.runAndWait()
            del speak_engine

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print final sentence at the end
print("\n📝 Final Sentence:", sentence)