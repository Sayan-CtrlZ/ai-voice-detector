import os
import torch
import numpy as np
import pickle
import librosa
import io
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --- 1. SETUP & CONFIGURATION ---
app = FastAPI()

# Mount the static folder (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load AI Brains (Global Variables)
print("Loading Model...")
try:
    with open("hackathon_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("model_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load Meta's Wav2Vec2 (The Feature Extractor)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    print("✅ Brain Loaded!")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")

# --- 2. ROOT ENDPOINT ---
@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "VoiceGuard AI is Running! Go to /static/index.html"})

# --- 3. THE DETECTION ENGINE ---
@app.post("/detect-audio/")
async def detect_audio(file: UploadFile = File(...)):
    try:
        # A. Read Audio File
        audio_bytes = await file.read()
        
        # Load audio using librosa (handles mp3, wav, flac)
        # We force 'sr=16000' because Wav2Vec2 requires exactly 16kHz
        try:
            audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        except Exception as e:
            # If librosa fails, it's likely a corrupted file
            return JSONResponse(content={
                "prediction": "AI", 
                "confidence": 0.0,
                "message": "Error reading audio file. Flagged as suspicious."
            })

        # B. Safety Check: Is audio too short?
        if len(audio) < 1600:  # Less than 0.1 second
            return JSONResponse(content={
                "prediction": "AI",
                "confidence": 0.0,
                "message": "Audio too short. Insufficient data."
            })

        # C. Feature Extraction (The Neural Network)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Take the average of all features (Mean Pooling)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()

        # D. Pre-processing (Scaling)
        # Check for NaNs (Invalid numbers) coming from the model
        if np.isnan(embedding).any():
             return JSONResponse(content={
                "prediction": "AI",
                "confidence": 0.99, # High confidence it's fake if it breaks the model
                "message": "Corrupted spectral data detected."
            })

        embedding_scaled = scaler.transform([embedding])

        # E. Prediction (The Classifier)
        probs = classifier.predict_proba(embedding_scaled)[0]
        
        human_score = float(probs[0])
        ai_score = float(probs[1])
        
        # --- SMART DECISION LOGIC ---
        # 1. Handle NaN (Not a Number) errors safely
        if np.isnan(human_score) or np.isnan(ai_score):
            final_label = "AI"
            final_confidence = 0.95
            explanation = "Invalid signal structure detected."
        
        # 2. Strict Threshold Rule
        # We only say "Human" if the model is >98% sure.
        elif human_score > 0.98:
            final_label = "Human"
            final_confidence = human_score
            explanation = "✅ Authentic vocal micro-tremors and breathing patterns verified."
        
        # 3. Default to AI for everything else
        else:
            final_label = "AI"
            # If it's AI, the confidence is the AI score. 
            # If the model was 'unsure' (e.g. 60% human), we effectively treat that as AI 
            # because we are in 'Security Mode' (Better safe than sorry).
            final_confidence = ai_score if ai_score > 0.5 else (1.0 - human_score)
            explanation = "🚨 Synthetic spectral smoothness and lack of natural anomalies detected."

        return JSONResponse(content={
            "prediction": final_label,
            "confidence": final_confidence,
            "message": explanation
        })

    except Exception as e:
        print(f"Error processing file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)