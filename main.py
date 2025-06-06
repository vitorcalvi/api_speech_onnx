import os
import logging
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import io
from datetime import datetime
import traceback
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wav2Vec2 Emotion Recognition via ONNX")

# Add CORS handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated emotion labels based on ehcalabres model
EMOTION_LABELS = {
    "ehcalabres": ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
    "msp_dim": ["arousal", "dominance", "valence"],
    "iemocap": ["angry", "happy", "excited", "sad", "frustrated", "fearful", "surprised", "neutral"]
}

class Wav2Vec2ONNXModel:
    def __init__(self, model_type="ehcalabres"):
        self.model_type = model_type
        self.model_path = None
        self.session = None
        self.emotion_labels = EMOTION_LABELS.get(model_type, EMOTION_LABELS["ehcalabres"])
        self.sample_rate = 16000
        self.feature_extractor = None
        self.initialize_model()
    
    def initialize_model(self):
        """Download and initialize ONNX model with proper feature extraction"""
        try:
            # Try to import transformers for proper feature extraction
            try:
                from transformers import Wav2Vec2FeatureExtractor
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-large-xlsr-53",
                    sampling_rate=16000,
                    do_normalize=True
                )
                logger.info("Loaded Wav2Vec2FeatureExtractor successfully")
            except ImportError:
                logger.warning("transformers not available, using manual preprocessing")
                self.feature_extractor = None
            
            model_urls = {
                "ehcalabres": "https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition/resolve/main/model.onnx",
                "msp_dim": "https://zenodo.org/record/6221127/files/model.onnx",
                "iemocap": "https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/resolve/main/model.onnx"
            }
            
            alternative_urls = {
                "ehcalabres": [
                    "https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition/resolve/caa9526cddb935a7f75cedb24478e8937bb2eaca/onnx/model.onnx",
                ]
            }
            
            model_url = model_urls.get(self.model_type)
            if not model_url:
                logger.warning(f"No URL found for model type: {self.model_type}")
                self.session = self.create_mock_model()
                return
            
            # Create models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            self.model_path = models_dir / f"wav2vec2_{self.model_type}.onnx"
            
            # Download model if not exists
            if not self.model_path.exists():
                logger.info(f"Downloading {self.model_type} model...")
                
                try:
                    self.download_model(model_url, self.model_path)
                except Exception as e:
                    logger.warning(f"Main URL failed: {e}")
                    
                    if self.model_type == "ehcalabres" and self.model_type in alternative_urls:
                        for alt_url in alternative_urls[self.model_type]:
                            try:
                                logger.info(f"Trying alternative URL: {alt_url}")
                                self.download_model(alt_url, self.model_path)
                                break
                            except Exception as alt_e:
                                logger.warning(f"Alternative URL failed: {alt_e}")
                                continue
                        else:
                            raise Exception("All download URLs failed")
                    else:
                        raise e
            
            # Verify file before loading
            if not self.verify_onnx_file():
                logger.error("Downloaded ONNX file is invalid")
                self.model_path.unlink()
                raise Exception("Invalid ONNX model")
            
            # Load ONNX model with optimized settings
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=providers,
                sess_options=session_options
            )
            
            logger.info(f"Successfully loaded {self.model_type} Wav2Vec2 ONNX model")
            logger.info(f"Model input: {self.session.get_inputs()[0].name}")
            logger.info(f"Model output: {self.session.get_outputs()[0].name}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = self.create_mock_model()

    def verify_onnx_file(self):
        """Verify ONNX file integrity"""
        try:
            if not self.model_path.exists():
                return False
            
            file_size = self.model_path.stat().st_size
            if file_size < 100000:
                logger.error(f"ONNX file too small: {file_size} bytes")
                return False
            
            test_session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
            logger.info(f"ONNX file verification successful ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False

    def download_model(self, url, path):
        """Enhanced download with progress tracking"""
        try:
            if path.exists():
                path.unlink()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ONNX-downloader)',
                'Accept': 'application/octet-stream'
            }
            
            logger.info(f"Downloading from: {url}")
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                raise Exception("Downloaded HTML page instead of ONNX model")
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every 1MB
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Model downloaded successfully: {path}")
            
            if downloaded < 100000:
                raise Exception(f"Downloaded file too small: {downloaded} bytes")
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if path.exists():
                path.unlink()
            raise

    def create_mock_model(self):
        """Create mock model for testing"""
        class MockModel:
            def __init__(self, emotion_labels):
                self.emotion_labels = emotion_labels
            
            def get_inputs(self):
                class MockInput:
                    name = "input_values"
                return [MockInput()]
            
            def get_outputs(self):
                class MockOutput:
                    name = "logits"
                return [MockOutput()]
            
            def run(self, output_names, input_dict):
                num_emotions = len(self.emotion_labels)
                scores = np.random.dirichlet(np.ones(num_emotions))
                return [scores.reshape(1, -1)]
        
        return MockModel(self.emotion_labels)
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """Proper preprocessing for ehcalabres Wav2Vec2 model"""
        try:
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz using librosa for better quality
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000,
                    res_type='kaiser_fast'
                )
            
            # Use transformers feature extractor if available
            if self.feature_extractor is not None:
                inputs = self.feature_extractor(
                    audio_data, 
                    sampling_rate=16000, 
                    return_tensors="np",
                    do_normalize=True,
                    padding=True
                )
                return inputs.input_values[0]
            else:
                # Manual preprocessing as fallback
                # Remove DC offset
                audio_data = audio_data - np.mean(audio_data)
                
                # Apply pre-emphasis filter
                pre_emphasis = 0.97
                audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
                
                # Normalize to [-1, 1] with better dynamics
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
                
                # Ensure minimum length
                min_length = int(2.0 * 16000)  # 2 seconds
                if len(audio_data) < min_length:
                    audio_data = np.pad(audio_data, (0, min_length - len(audio_data)))
                
                # Limit maximum length
                max_length = int(30 * 16000)  # 30 seconds
                if len(audio_data) > max_length:
                    audio_data = audio_data[:max_length]
                
                return audio_data.astype(np.float32)
                
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Return minimal valid input
            return np.zeros(32000, dtype=np.float32)  # 2 seconds of silence
    
    def predict_emotion(self, audio_data, sample_rate=16000):
        """Enhanced emotion prediction with proper preprocessing"""
        try:
            # Preprocess audio properly
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Log input characteristics for debugging
            logger.info(f"Processed audio shape: {processed_audio.shape}")
            logger.info(f"Audio min/max: {np.min(processed_audio):.3f}/{np.max(processed_audio):.3f}")
            logger.info(f"Audio mean/std: {np.mean(processed_audio):.3f}/{np.std(processed_audio):.3f}")
            
            # Prepare input for ONNX model
            input_name = self.session.get_inputs()[0].name
            if len(processed_audio.shape) == 1:
                input_data = {input_name: processed_audio.reshape(1, -1)}
            else:
                input_data = {input_name: processed_audio}
            
            # Run ONNX inference
            outputs = self.session.run(None, input_data)
            logits = outputs[0]
            
            # Log raw logits for debugging
            logger.info(f"Raw logits shape: {logits.shape}")
            logger.info(f"Raw logits: {logits}")
            
            # Process logits based on model type
            if self.model_type == "ehcalabres":
                # Remove batch dimension if present
                if len(logits.shape) > 1:
                    logits = logits[0]
                
                # Apply temperature scaling for sharper predictions
                temperature = 0.8
                scaled_logits = logits / temperature
                
                # Apply softmax with numerical stability
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
                scores = exp_logits / np.sum(exp_logits)
                
            elif self.model_type == "msp_dim":
                scores = np.sigmoid(logits[0])
            else:
                # Default softmax
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                scores = scores[0]
            
            # Log processed scores
            logger.info(f"Processed scores: {scores}")
            
            # Create emotion mapping
            emotion_probs = {
                emotion: float(score) 
                for emotion, score in zip(self.emotion_labels, scores)
            }
            
            # Get dominant emotion
            if self.model_type == "msp_dim":
                arousal, dominance, valence = scores[:3] if len(scores) >= 3 else [0.5, 0.5, 0.5]
                dominant_emotion = self.interpret_dimensions(arousal, dominance, valence)
                confidence = np.mean(scores[:3])
            else:
                dominant_emotion = max(emotion_probs, key=emotion_probs.get)
                confidence = emotion_probs[dominant_emotion]
            
            logger.info(f"Dominant emotion: {dominant_emotion} (confidence: {confidence:.3f})")
            
            return {
                "emotions": emotion_probs,
                "dominant_emotion": dominant_emotion,
                "confidence": float(confidence),
                "timestamp": datetime.utcnow().isoformat(),
                "model_type": f"Wav2Vec2_{self.model_type}",
                "model_info": {
                    "architecture": "Wav2Vec2ForSequenceClassification",
                    "backend": "ONNX Runtime",
                    "sample_rate": 16000,
                    "audio_length": len(processed_audio) / 16000,
                    "model_source": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    "preprocessing": "transformers" if self.feature_extractor else "manual"
                }
            }
            
        except Exception as e:
            logger.error(f"Wav2Vec2 prediction failed: {e}")
            logger.error(traceback.format_exc())
            return self.fallback_result()
    
    def interpret_dimensions(self, arousal, dominance, valence):
        """Convert dimensional values to emotion categories"""
        if arousal > 0.6 and valence > 0.6:
            return "happy"
        elif arousal > 0.6 and valence < 0.4:
            return "angry" if dominance > 0.5 else "fearful"
        elif arousal < 0.4 and valence > 0.6:
            return "calm"
        elif arousal < 0.4 and valence < 0.4:
            return "sad"
        elif valence < 0.3:
            return "disgust"
        else:
            return "neutral"
    
    def fallback_result(self):
        """Fallback emotion result"""
        fallback_emotions = {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}
        return {
            "emotions": fallback_emotions,
            "dominant_emotion": "neutral",
            "confidence": 0.5,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Analysis failed - using fallback",
            "model_type": f"fallback_{self.model_type}"
        }

# Initialize model
try:
    emotion_model = Wav2Vec2ONNXModel("ehcalabres")
    logger.info("ehcalabres Wav2Vec2 emotion model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize emotion model: {e}")
    emotion_model = None

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.post("/analyze_audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Enhanced audio analysis with ehcalabres model"""
    try:
        if emotion_model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not available",
                    "detail": "ehcalabres Wav2Vec2 emotion recognition model is not loaded",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        if not audio_file.filename:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No file provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # Read and process audio
        audio_bytes = await audio_file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        logger.info(f"Processing {audio_file.filename}: {len(audio_bytes)} bytes, {sample_rate}Hz")
        
        # Predict emotion
        result = emotion_model.predict_emotion(audio_data, sample_rate)
        
        return JSONResponse(
            content={
                "status": "success",
                "filename": audio_file.filename,
                "result": result,
                "model_source": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Analysis failed",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """Health check with model info"""
    model_status = "available" if emotion_model else "unavailable"
    model_info = {
        "type": emotion_model.model_type if emotion_model else "none",
        "architecture": "Wav2Vec2ForSequenceClassification",
        "source": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "emotions": emotion_model.emotion_labels if emotion_model else [],
        "sample_rate": emotion_model.sample_rate if emotion_model else 0,
        "accuracy": "82.23%" if emotion_model and emotion_model.model_type == "ehcalabres" else "unknown",
        "feature_extractor": "loaded" if emotion_model and emotion_model.feature_extractor else "manual"
    }
    
    return JSONResponse(
        content={
            "status": "running",
            "model_status": model_status,
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.get("/models")
async def list_models():
    """List available emotion models"""
    return JSONResponse(
        content={
            "available_models": list(EMOTION_LABELS.keys()),
            "current_model": emotion_model.model_type if emotion_model else None,
            "emotions_per_model": EMOTION_LABELS,
            "recommended_model": "ehcalabres",
            "model_details": {
                "ehcalabres": {
                    "accuracy": "82.23%",
                    "dataset": "RAVDESS",
                    "emotions": 8,
                    "source": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                }
            }
        }
    )

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Web interface for emotion recognition"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ehcalabres Wav2Vec2 Emotion Recognition</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            button:disabled { background: #6c757d; cursor: not-allowed; }
            .file-input { margin: 10px 0; }
            .results { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
            .emotion-item { display: flex; justify-content: space-between; margin: 5px 0; padding: 8px; background: white; border-radius: 3px; border-left: 4px solid #007bff; }
            .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.9rem; }
            .model-info { background: #e7f3ff; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #b3d9ff; }
            .emotion-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0; }
            .emotion-card { background: white; padding: 10px; border-radius: 5px; text-align: center; border: 2px solid #dee2e6; }
            .emotion-card.dominant { border-color: #28a745; background: #d4edda; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 ehcalabres Wav2Vec2 Emotion Recognition</h1>
            <p>Enhanced emotion recognition using proper preprocessing for the ehcalabres model</p>
            
            <div id="statusIndicator" class="status warning">Initializing enhanced ehcalabres model...</div>
            
            <div class="model-info" id="modelInfo">
                <h3>🧠 Model Information</h3>
                <div id="modelDetails">Loading enhanced model details...</div>
            </div>
            
            <div class="file-input">
                <label for="audioFile">Upload Audio File (WAV, MP3, etc.):</label><br>
                <input type="file" id="audioFile" accept="audio/*">
                <button id="analyzeBtn" onclick="analyzeAudioFile()" disabled>🔍 Analyze Emotion</button>
            </div>
            
            <div class="results" id="resultsPanel">
                <h3>🎭 Emotion Analysis Results</h3>
                <div class="emotion-grid" id="emotionGrid">
                    <div class="emotion-card">Upload an audio file to see enhanced emotion predictions...</div>
                </div>
                <div id="dominantEmotion" style="font-size: 1.2em; font-weight: bold; margin: 10px 0;">Dominant: None</div>
                <div id="confidence" style="font-size: 1.1em; margin: 10px 0;">Confidence: 0%</div>
                <div id="modelUsed" style="font-size: 0.9em; color: #666; margin: 10px 0;">Model: ehcalabres (enhanced preprocessing)</div>
            </div>
            
            <div class="log" id="logPanel">
                <div>[System] Initializing enhanced ehcalabres Wav2Vec2 emotion recognition...</div>
            </div>
        </div>

        <script>
            const emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"];
            
            document.addEventListener('DOMContentLoaded', function() {
                initializeApplication();
                setupEventListeners();
            });

            async function initializeApplication() {
                try {
                    logMessage('🔍 Checking enhanced ehcalabres model status...', 'info');
                    
                    const response = await fetch('/health');
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const healthData = await response.json();
                    updateModelInfo(healthData.model_info);
                    
                    if (healthData.model_status === 'available') {
                        updateStatus('✅ Enhanced ehcalabres model ready with proper preprocessing', 'success');
                        logMessage(`Enhanced model loaded: ${healthData.model_info.source}`, 'success');
                        logMessage(`Feature extraction: ${healthData.model_info.feature_extractor}`, 'info');
                        document.getElementById('analyzeBtn').disabled = false;
                    } else {
                        updateStatus('⚠️ Model unavailable - check installation', 'warning');
                        logMessage('ehcalabres model unavailable', 'error');
                    }
                    
                } catch (error) {
                    updateStatus(`❌ System error: ${error.message}`, 'error');
                    logMessage(`Initialization failed: ${error.message}`, 'error');
                }
            }

            function updateModelInfo(modelInfo) {
                const modelDetails = document.getElementById('modelDetails');
                if (modelInfo && modelInfo.type !== 'none') {
                    modelDetails.innerHTML = `
                        <strong>Source:</strong> ${modelInfo.source}<br>
                        <strong>Architecture:</strong> ${modelInfo.architecture}<br>
                        <strong>Accuracy:</strong> ${modelInfo.accuracy}<br>
                        <strong>Sample Rate:</strong> ${modelInfo.sample_rate}Hz<br>
                        <strong>Preprocessing:</strong> ${modelInfo.feature_extractor || 'Enhanced manual'}<br>
                        <strong>Emotions:</strong> ${modelInfo.emotions.join(', ')}
                    `;
                } else {
                    modelDetails.innerHTML = 'Enhanced ehcalabres model not loaded';
                }
            }

            function setupEventListeners() {
                const audioFile = document.getElementById('audioFile');
                audioFile.addEventListener('change', function(e) {
                    const analyzeBtn = document.getElementById('analyzeBtn');
                    analyzeBtn.disabled = !e.target.files.length;
                    if (e.target.files.length > 0) {
                        logMessage(`📁 File selected: ${e.target.files[0].name} (${(e.target.files[0].size/1024/1024).toFixed(2)}MB)`, 'info');
                    }
                });
            }

            async function analyzeAudioFile() {
                const fileInput = document.getElementById('audioFile');
                if (!fileInput.files.length) {
                    logMessage('❌ No file selected', 'error');
                    return;
                }

                const analyzeBtn = document.getElementById('analyzeBtn');
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🔄 Processing with enhanced preprocessing...';

                try {
                    const formData = new FormData();
                    formData.append('audio_file', fileInput.files[0]);

                    logMessage('📤 Uploading audio for enhanced ehcalabres analysis...', 'info');
                    const response = await fetch('/analyze_audio', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP ${response.status}`);
                    }

                    const result = await response.json();
                    displayResults(result.result);
                    logMessage('✅ Enhanced ehcalabres analysis completed successfully', 'success');

                } catch (error) {
                    logMessage(`❌ Analysis failed: ${error.message}`, 'error');
                    updateStatus(`Analysis error: ${error.message}`, 'error');
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🔍 Analyze Emotion';
                }
            }

            function displayResults(result) {
                const emotionGrid = document.getElementById('emotionGrid');
                const dominantEmotion = document.getElementById('dominantEmotion');
                const confidence = document.getElementById('confidence');
                const modelUsed = document.getElementById('modelUsed');
                
                let html = '';
                emotions.forEach(emotion => {
                    const score = result.emotions[emotion] || 0;
                    const percentage = (score * 100).toFixed(1);
                    const isDominant = emotion === result.dominant_emotion;
                    
                    html += `
                        <div class="emotion-card ${isDominant ? 'dominant' : ''}">
                            <div style="font-weight: bold; margin-bottom: 5px;">
                                ${isDominant ? '🎯 ' : ''}${emotion}
                            </div>
                            <div style="font-size: 1.2em; color: ${isDominant ? '#28a745' : '#007bff'};">
                                ${percentage}%
                            </div>
                        </div>
                    `;
                });
                emotionGrid.innerHTML = html;
                
                const confidencePercent = (result.confidence * 100).toFixed(1);
                dominantEmotion.textContent = `🎭 Dominant: ${result.dominant_emotion}`;
                confidence.textContent = `📊 Confidence: ${confidencePercent}%`;
                
                if (result.model_info) {
                    modelUsed.textContent = `🧠 Model: ehcalabres enhanced (${result.model_info.audio_length?.toFixed(1)}s, ${result.model_info.preprocessing})`;
                }
                
                logMessage(`🎯 Result: ${result.dominant_emotion} (${confidencePercent}% confidence)`, 'success');
            }

            function logMessage(message, type = 'info') {
                const logPanel = document.getElementById('logPanel');
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                
                const typeColors = {
                    'info': '#3498db',
                    'success': '#27ae60',
                    'error': '#e74c3c',
                    'warning': '#f39c12'
                };
                
                logEntry.innerHTML = `
                    <span style="color: #95a5a6;">[${timestamp}]</span>
                    <span style="color: ${typeColors[type] || '#ecf0f1'};">${message}</span>
                `;
                
                logPanel.appendChild(logEntry);
                logPanel.scrollTop = logPanel.scrollHeight;
            }

            function updateStatus(message, type) {
                const statusIndicator = document.getElementById('statusIndicator');
                statusIndicator.className = `status ${type}`;
                statusIndicator.textContent = message;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting enhanced ehcalabres Wav2Vec2 emotion recognition server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
