import os
import glob
import numpy as np
import librosa
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --- 1. Setup Models ---
print("Step 1: Loading AI Model... (This will download ~300MB the first time)")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_features(file_path):
    """Loads audio and extracts the deep voice print."""
    # librosa.load works with mp3 and m4a if ffmpeg is installed
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Process with Wav2Vec2
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate the average voice feature
    last_hidden_states = outputs.last_hidden_state
    feature_vector = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
    return feature_vector

# --- 2. Load Data ---
X = [] # Features (The voice numbers)
y = [] # Labels (0 = Real, 1 = AI)

# Helper to find mp3 AND m4a files
def get_files(folder):
    return glob.glob(f"{folder}/*.mp3") + glob.glob(f"{folder}/*.wav")

print("\nStep 2: Processing Real Audio...")
real_files = get_files("dataset/real")
for file in real_files:
    print(f"  - Loading {file}...")
    try:
        feat = extract_features(file)
        X.append(feat)
        y.append(0) # 0 = Human
    except Exception as e:
        print(f"    [!] Error loading {file}. Is FFmpeg installed? Error: {e}")

print("\nStep 3: Processing AI Audio...")
ai_files = get_files("dataset/ai")
for file in ai_files:
    print(f"  - Loading {file}...")
    try:
        feat = extract_features(file)
        X.append(feat)
        y.append(1) # 1 = AI
    except Exception as e:
        print(f"    [!] Error loading {file}: {e}")

# --- 3. Train & Save ---
if len(X) == 0:
    print("\n[ERROR] No files were loaded. Check your folders or install FFmpeg.")
else:
    print(f"\nStep 4: Training Classifier on {len(X)} files...")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    # Save the brain to a file
    joblib.dump(clf, "hackathon_model.pkl")
    print("\nSUCCESS! Model saved as 'hackathon_model.pkl'. You are ready for the backend!")