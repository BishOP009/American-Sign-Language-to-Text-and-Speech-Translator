import os
import cv2
import numpy as np
import faiss
import pickle
import mediapipe as mp
from tqdm import tqdm

# -----------------------
# Configuration
# -----------------------
DATA_DIR = "F:/Sign Language Converter/dataset/asl_alphabet_train"  # Your full dataset path
SAVE_DIR = "F:/Sign Language Converter/index_labels"   # Folder to save index and mapping
os.makedirs(SAVE_DIR, exist_ok=True)

INDEX_FILE = os.path.join(SAVE_DIR, "faiss_index_ivfflat.index")
MAPPING_FILE = os.path.join(SAVE_DIR, "index_to_label.pkl")

EMBEDDING_DIM = 63  # 21 landmarks * 3 coordinates
nlist = 10          # Number of IVF clusters

# -----------------------
# Mediapipe setup
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

# -----------------------
# Step 1: Function to extract hand landmarks
# -----------------------
def get_hand_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        embedding = []
        for lm in hand_landmarks.landmark:
            embedding.extend([lm.x, lm.y, lm.z])
        return np.array(embedding, dtype=np.float32)
    else:
        return None

# -----------------------
# Step 2: Load images and generate embeddings with progress bar
# -----------------------
embeddings = []
labels = []

# Count total images for progress bar
total_images = sum(len(files) for r, d, files in os.walk(DATA_DIR))
pbar = tqdm(total=total_images, desc="Processing images")

for sign_name in os.listdir(DATA_DIR):
    sign_path = os.path.join(DATA_DIR, sign_name)
    if not os.path.isdir(sign_path):
        continue

    for img_name in os.listdir(sign_path):
        img_path = os.path.join(sign_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            pbar.update(1)
            continue

        emb = get_hand_embedding(image)
        if emb is not None:
            embeddings.append(emb)
            labels.append(sign_name)
        pbar.update(1)

pbar.close()

embeddings = np.array(embeddings).astype('float32')
labels = np.array(labels)

print(f"Total embeddings: {embeddings.shape[0]}")

# -----------------------
# Step 2.5: L2 Normalization
# -----------------------
faiss.normalize_L2(embeddings)   # In-place normalization
# -----------------------
# Step 3: Create IndexIVFFlat
# -----------------------
quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_L2)

# Step 4: Train the index
print("Training FAISS index...")
index.train(embeddings)
print("Training complete.")

# Step 5: Add embeddings to the index
index.add(embeddings)
print(f"Total vectors added to index: {index.ntotal}")

# Step 6: Save index and labels
faiss.write_index(index, INDEX_FILE)
with open(MAPPING_FILE, "wb") as f:
    pickle.dump(labels.tolist(), f)

print("FAISS index and labels saved successfully.")