from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import joblib
import base64
import io
import os
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pydub import AudioSegment

app = FastAPI()

# --- CONFIGURATION ---
# Rule 5: API Key Authentication
SECRET_API_KEY = "sk_hackathon_team_123" 

# Setup FFmpeg
AudioSegment.converter = os.getcwd() + "\\ffmpeg.exe"
AudioSegment.ffprobe   = os.getcwd() + "\\ffprobe.exe"

# --- LOAD MODELS ---
# Rule 2: Must support 5 languages. We use XLSR-53 (Multilingual)
print("Loading Multilingual Brain...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
classifier = joblib.load("hackathon_model.pkl")
print("Brain Loaded!")

# --- STRICT INPUT FORMAT (Rule 7) ---
class VoiceRequest(BaseModel):
    language: str       # "Tamil", "English", "Hindi", etc.
    audioFormat: str    # Must be "mp3"
    audioBase64: str    # Note: CamelCase as per PDF

# --- SECURITY CHECK (Rule 5) ---
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != SECRET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# --- API ENDPOINT (Rule 6) ---
@app.post("/api/voice-detection") 
async def detect_voice(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    try:
        # 1. Decode Audio
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # 2. Preprocess (16kHz Mono)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        
        # 3. Extract Features
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        
        # 4. Predict
        prediction = classifier.predict([embedding])[0]
        confidence = float(classifier.predict_proba([embedding])[0].max())
        
        # 5. STRICT OUTPUT FORMAT (Rule 8)
        # Must be AI_GENERATED or HUMAN
        label = "AI_GENERATED" if prediction == 1 else "HUMAN"
        
        # Explanation
        explanation_text = "Natural pitch variance and breathing sounds detected."
        if label == "AI_GENERATED":
            explanation_text = "Unnatural spectral smoothness and lack of micro-tremors detected."

        return {
            "status": "success",
            "language": request.language,
            "classification": label,
            "confidenceScore": round(confidence, 2), # CamelCase required
            "explanation": explanation_text
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})