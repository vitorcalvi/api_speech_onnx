from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import tempfile
import os
from functools import lru_cache

app = FastAPI(title="Speech Emotion Recognition API")

# Global variables for model caching
model = None
feature_extractor = None
id2label = None

@lru_cache(maxsize=1)
def load_model():
    """Load model once and cache it for CPU optimization"""
    global model, feature_extractor, id2label
    
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    
    # Force CPU usage for free tier
    device = "cpu"
    torch.set_num_threads(2)  # Optimize for free CPU
    
    model = AutoModelForAudioClassification.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, 
        do_normalize=True
    )
    id2label = model.config.id2label
    
    return model, feature_extractor, id2label

def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    """Preprocess audio with memory optimization"""
    audio_array, sampling_rate = librosa.load(
        audio_path, 
        sr=feature_extractor.sampling_rate,
        duration=max_duration  # Limit duration for CPU efficiency
    )
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/predict-emotion")
async def predict_emotion(file: UploadFile = File(...)):
    """Predict emotion from uploaded audio file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load cached model
            model, feature_extractor, id2label = load_model()
            
            # Preprocess and predict
            inputs = preprocess_audio(tmp_file_path, feature_extractor)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_id = torch.argmax(logits, dim=-1).item()
                predicted_label = id2label[predicted_id]
                
                # Get confidence scores
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities[0][predicted_id].item()
            
            return {
                "predicted_emotion": predicted_label,
                "confidence": round(confidence, 4),
                "all_emotions": {
                    id2label[i]: round(probabilities[0][i].item(), 4) 
                    for i in range(len(id2label))
                }
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Speech Emotion Recognition API",
        "model": "Whisper Large V3",
        "emotions": ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"],
        "endpoints": {
            "predict": "/predict-emotion",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
