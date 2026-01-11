🤟 American Sign Language (ASL) to Text and Speech Translator

A real-time computer vision–based web application that translates American Sign Language (ASL) hand gestures into text and spoken output. The system uses MediaPipe hand landmarks and FAISS-based similarity search to recognize gestures through a live webcam feed, presented via an interactive Streamlit interface.

🚀 Features
- Real-time ASL alphabet recognition using a webcam
- Converts hand signs into text, building sentences letter by letter
- Offline text-to-speech conversion for the recognized sentence
- Non-blocking speech (camera feed continues during audio playback)
- Checkbox-based controls for camera and recognition
- Clean and interactive Streamlit web interface
- Fully offline execution after setup
- Suitable for academic, portfolio, and demonstration purposes

🧠 Tech Stack
Language: Python  
Computer Vision: OpenCV, MediaPipe  
Vector Search: FAISS  
Numerical Computing: NumPy  
Text-to-Speech: pyttsx3  
Web Framework: Streamlit  

📁 Project Structure
ASL-Translator/
│
├── app.py                         Streamlit application
├── requirements.txt               Project dependencies
├── index_labels/
│   ├── faiss_index_ivfflat.index  FAISS index for gesture embeddings
│   └── index_to_label.pkl         Index-to-label mapping
├── dataset/
│   └── asl_alphabet_train/        ASL alphabet dataset
└── README.md

⚙️ Installation and Setup

1. Clone the repository
git clone <repository-url>
cd ASL-Translator

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate
Windows users: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py

The application will be available at:
http://localhost:8501

🖐️ How It Works
- MediaPipe extracts 3D hand landmark coordinates from the webcam feed
- Landmarks are converted into embeddings and normalized
- FAISS performs fast nearest-neighbor search to classify the gesture
- Stable gestures are appended to the sentence after a short hold duration
- The final sentence can be spoken using offline text-to-speech

🔐 Input & Interaction Notes
- Recognition starts only when the camera and recognition checkboxes are enabled
- Sentence can be cleared at any time while the camera is running
- The system ignores invalid or unstable gestures to improve accuracy

📌 Disclaimer
This project is intended for educational and demonstration purposes only. It is not a certified sign language translation system and may not cover all ASL grammar or vocabulary.

👤 Author
Ayanava Kundu  
Computer Science and Engineering (Data Science)  
AI/ML | Data Science | Software Development
