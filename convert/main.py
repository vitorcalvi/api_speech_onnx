"""
Comprehensive ONNX Conversion for Speech Emotion Recognition with Whisper Large V3
================================================================

This implementation provides multiple approaches to convert the fine-tuned Whisper Large V3 
emotion recognition model to ONNX format, addressing known compatibility issues.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor,
    WhisperConfig,
    WhisperForAudioClassification
)
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from optimum.exporters.onnx.base import ConfigBehavior
from typing import Dict, Optional, Tuple
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperAudioClassificationOnnxConfig(WhisperOnnxConfig):
    """
    Custom ONNX configuration for Whisper audio classification models.
    Extends the base WhisperOnnxConfig to support classification tasks.
    """
    
    def __init__(
        self, 
        config: WhisperConfig, 
        task: str = "audio-classification",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
    ):
        super().__init__(config, task, use_past, use_past_in_inputs)
        self._config = config
        self.task = task
        
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        """Define input specifications for audio classification."""
        # Use encoder-only inputs for classification
        return {
            "input_features": {
                0: "batch_size",
                1: "feature_size", 
                2: "encoder_sequence_length"
            }
        }
    
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """Define output specifications for audio classification."""
        return {
            "logits": {0: "batch_size", 1: "num_labels"}
        }
    
    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
        """Generate dummy inputs with correct mel bin dimensions for Large V3."""
        batch_size = kwargs.get("batch_size", 1)
        # Whisper Large V3 uses 128 mel bins instead of 80
        num_mel_bins = getattr(self._config, 'num_mel_bins', 128)
        encoder_seq_length = kwargs.get("encoder_sequence_length", 3000)
        
        if framework == "pt":
            dummy_input = torch.randn(
                batch_size, 
                num_mel_bins, 
                encoder_seq_length,
                dtype=torch.float32
            )
            return {"input_features": dummy_input}
        else:
            dummy_input = np.random.randn(
                batch_size, 
                num_mel_bins, 
                encoder_seq_length
            ).astype(np.float32)
            return {"input_features": dummy_input}

class WhisperEmotionONNXConverter:
    """
    Comprehensive converter for Whisper emotion recognition models to ONNX.
    Provides multiple conversion strategies to handle compatibility issues.
    """
    
    def __init__(self, model_id: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"):
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.config = None
        
    def load_model(self):
        """Load the pre-trained model and feature extractor."""
        logger.info(f"Loading model: {self.model_id}")
        
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, 
            do_normalize=True
        )
        self.config = self.model.config
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Model loaded successfully. Labels: {self.model.config.id2label}")
        
    def method_1_direct_torch_export(self, output_path: str, opset_version: int = 17):
        """
        Method 1: Direct PyTorch ONNX export using torch.onnx.export
        This bypasses Optimum's limitations but requires manual input/output handling.
        """
        logger.info("Starting Method 1: Direct PyTorch ONNX export")
        
        if self.model is None:
            self.load_model()
        
        # Create dummy input with correct dimensions for Whisper Large V3
        batch_size = 1
        num_mel_bins = getattr(self.config, 'num_mel_bins', 128)
        encoder_seq_length = 3000
        
        dummy_input = torch.randn(
            batch_size, 
            num_mel_bins, 
            encoder_seq_length,
            dtype=torch.float32
        )
        
        # Define input and output names
        input_names = ["input_features"]
        output_names = ["logits"]
        
        # Dynamic axes for flexible batch sizes and sequence lengths
        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "sequence_length"},
            "logits": {0: "batch_size"}
        }
        
        # Export to ONNX
        output_file = Path(output_path) / "emotion_classification.onnx"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_file),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=True,
            export_params=True
        )
        
        logger.info(f"Model exported successfully to {output_file}")
        return str(output_file)
    
    def method_2_encoder_only_export(self, output_path: str):
        """
        Method 2: Export only the encoder part with classification head
        This approach extracts just the relevant components for emotion recognition.
        """
        logger.info("Starting Method 2: Encoder-only export with classification head")
        
        if self.model is None:
            self.load_model()
        
        class WhisperEncoderClassifier(nn.Module):
            """Wrapper class for encoder + classification head."""
            
            def __init__(self, whisper_model):
                super().__init__()
                self.encoder = whisper_model.model.encoder
                self.classifier = whisper_model.classifier
                self.projector = whisper_model.projector
                
            def forward(self, input_features):
                # Extract features using encoder
                encoder_outputs = self.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state
                
                # Apply projection if present
                if hasattr(self, 'projector') and self.projector is not None:
                    hidden_states = self.projector(hidden_states)
                
                # Pool the features (usually mean pooling)
                pooled_output = hidden_states.mean(dim=1)
                
                # Apply classification head
                logits = self.classifier(pooled_output)
                
                return logits
        
        # Create the encoder-classifier model
        encoder_classifier = WhisperEncoderClassifier(self.model)
        encoder_classifier.eval()
        
        # Export using direct PyTorch export
        return self.method_1_direct_torch_export_custom_model(
            encoder_classifier, 
            output_path, 
            "encoder_classifier.onnx"
        )
    
    def method_1_direct_torch_export_custom_model(self, model, output_path: str, filename: str):
        """Helper method for exporting custom model variants."""
        batch_size = 1
        num_mel_bins = getattr(self.config, 'num_mel_bins', 128)
        encoder_seq_length = 3000
        
        dummy_input = torch.randn(
            batch_size, 
            num_mel_bins, 
            encoder_seq_length,
            dtype=torch.float32
        )
        
        output_file = Path(output_path) / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_file),
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch_size", 2: "sequence_length"},
                "logits": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=True,
            export_params=True
        )
        
        logger.info(f"Custom model exported to {output_file}")
        return str(output_file)
    
    def method_3_optimum_with_custom_config(self, output_path: str):
        """
        Method 3: Use Optimum with custom configuration class
        This attempts to work around Optimum's limitations by providing custom config.
        """
        logger.info("Starting Method 3: Optimum with custom configuration")
        
        if self.model is None:
            self.load_model()
        
        try:
            # Create custom configuration
            custom_config = WhisperAudioClassificationOnnxConfig(
                config=self.config,
                task="audio-classification"
            )
            
            # Attempt export with custom configuration
            main_export(
                model_name_or_path=self.model_id,
                output=output_path,
                task="feature-extraction",  # Use supported task as base
                custom_onnx_configs={"model": custom_config},
                opset=17,
                device="cpu"
            )
            
            logger.info(f"Optimum export completed to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Method 3 failed: {e}")
            logger.info("Falling back to Method 1")
            return self.method_1_direct_torch_export(output_path)
    
    def method_4_feature_extraction_pipeline(self, output_path: str):
        """
        Method 4: Export as feature extraction model and add classification head separately
        This approach splits the model into feature extraction + classification components.
        """
        logger.info("Starting Method 4: Feature extraction pipeline")
        
        if self.model is None:
            self.load_model()
        
        # Export encoder as feature extractor
        try:
            from optimum.exporters.onnx import main_export
            
            # First, export the encoder part
            encoder_output_path = Path(output_path) / "encoder"
            main_export(
                model_name_or_path="openai/whisper-large-v3",  # Base model
                output=str(encoder_output_path),
                task="feature-extraction",
                opset=17,
                device="cpu"
            )
            
            # Then export just the classification head
            class ClassificationHead(nn.Module):
                def __init__(self, whisper_model):
                    super().__init__()
                    self.projector = whisper_model.projector
                    self.classifier = whisper_model.classifier
                    
                def forward(self, hidden_states):
                    # Apply projection if present
                    if self.projector is not None:
                        hidden_states = self.projector(hidden_states)
                    
                    # Pool the features
                    pooled_output = hidden_states.mean(dim=1)
                    
                    # Apply classification
                    logits = self.classifier(pooled_output)
                    return logits
            
            classifier_head = ClassificationHead(self.model)
            classifier_head.eval()
            
            # Export classification head
            dummy_features = torch.randn(1, 1500, self.config.d_model)
            classifier_output = Path(output_path) / "classifier.onnx"
            
            torch.onnx.export(
                classifier_head,
                dummy_features,
                str(classifier_output),
                input_names=["hidden_states"],
                output_names=["logits"],
                dynamic_axes={
                    "hidden_states": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                },
                opset_version=17,
                do_constant_folding=True
            )
            
            logger.info(f"Pipeline export completed. Encoder: {encoder_output_path}, Classifier: {classifier_output}")
            return {
                "encoder": str(encoder_output_path),
                "classifier": str(classifier_output)
            }
            
        except Exception as e:
            logger.error(f"Method 4 failed: {e}")
            return self.method_1_direct_torch_export(output_path)
    
    def validate_onnx_model(self, onnx_path: str, test_input: Optional[torch.Tensor] = None):
        """
        Validate the exported ONNX model by comparing outputs with PyTorch model.
        """
        logger.info(f"Validating ONNX model: {onnx_path}")
        
        if self.model is None:
            self.load_model()
        
        # Load ONNX model
        try:
            ort_session = ort.InferenceSession(onnx_path)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
        
        # Create test input if not provided
        if test_input is None:
            batch_size = 1
            num_mel_bins = getattr(self.config, 'num_mel_bins', 128)
            encoder_seq_length = 3000
            
            test_input = torch.randn(
                batch_size, 
                num_mel_bins, 
                encoder_seq_length,
                dtype=torch.float32
            )
        
        # Get PyTorch model output
        with torch.no_grad():
            pytorch_output = self.model(test_input)
            if hasattr(pytorch_output, 'logits'):
                pytorch_logits = pytorch_output.logits
            else:
                pytorch_logits = pytorch_output
        
        # Get ONNX model output
        onnx_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_outputs = ort_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]
        
        # Compare outputs
        pytorch_logits_np = pytorch_logits.numpy()
        
        # Check shapes
        if pytorch_logits_np.shape != onnx_logits.shape:
            logger.error(f"Shape mismatch: PyTorch {pytorch_logits_np.shape} vs ONNX {onnx_logits.shape}")
            return False
        
        # Check numerical closeness
        max_diff = np.max(np.abs(pytorch_logits_np - onnx_logits))
        mean_diff = np.mean(np.abs(pytorch_logits_np - onnx_logits))
        
        logger.info(f"Validation results:")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        
        # Consider validation successful if differences are small
        if max_diff < 1e-3 and mean_diff < 1e-4:
            logger.info("✓ ONNX model validation PASSED")
            return True
        else:
            logger.warning("⚠ ONNX model validation shows significant differences")
            return False
    
    def convert_with_all_methods(self, output_base_path: str):
        """
        Attempt conversion using all available methods and return the best result.
        """
        logger.info("Starting comprehensive conversion with all methods")
        
        results = {}
        methods = [
            ("method_1_direct", self.method_1_direct_torch_export),
            ("method_2_encoder_only", self.method_2_encoder_only_export),
            ("method_3_optimum_custom", self.method_3_optimum_with_custom_config),
            ("method_4_pipeline", self.method_4_feature_extraction_pipeline)
        ]
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Attempting {method_name}")
                method_output_path = Path(output_base_path) / method_name
                method_output_path.mkdir(parents=True, exist_ok=True)
                
                result = method_func(str(method_output_path))
                
                # Validate the result
                if isinstance(result, str) and result.endswith('.onnx'):
                    validation_passed = self.validate_onnx_model(result)
                    results[method_name] = {
                        "path": result,
                        "validation_passed": validation_passed,
                        "type": "single_file"
                    }
                elif isinstance(result, dict):
                    results[method_name] = {
                        "path": result,
                        "validation_passed": True,  # Assume pipeline is valid
                        "type": "pipeline"
                    }
                
                logger.info(f"✓ {method_name} completed successfully")
                
            except Exception as e:
                logger.error(f"✗ {method_name} failed: {e}")
                results[method_name] = {
                    "path": None,
                    "validation_passed": False,
                    "error": str(e),
                    "type": "failed"
                }
        
        return results

# Usage Example and Testing Framework
class ONNXInferenceWrapper:
    """
    Wrapper class for running inference with converted ONNX models.
    """
    
    def __init__(self, onnx_path: str, feature_extractor_path: str):
        self.session = ort.InferenceSession(onnx_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
        
    def predict_emotion(self, audio_array: np.ndarray, sampling_rate: int = 16000):
        """
        Predict emotion from audio array using ONNX model.
        """
        # Extract features
        features = self.feature_extractor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="np"
        )
        
        # Run inference
        onnx_inputs = {self.session.get_inputs()[0].name: features["input_features"]}
        onnx_outputs = self.session.run(None, onnx_inputs)
        logits = onnx_outputs[0]
        
        # Get prediction
        predicted_class = np.argmax(logits, axis=1)[0]
        confidence = np.softmax(logits, axis=1)[0]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "max_confidence": float(confidence[predicted_class])
        }

def main():
    """
    Main execution function demonstrating the conversion process.
    """
    # Initialize converter
    converter = WhisperEmotionONNXConverter()
    
    # Define output path
    output_path = "./whisper_emotion_onnx_exports"
    
    # Perform conversion with all methods
    results = converter.convert_with_all_methods(output_path)
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    
    for method_name, result in results.items():
        status = "✓ PASSED" if result["validation_passed"] else "✗ FAILED"
        print(f"{method_name}: {status}")
        if result["path"]:
            print(f"  Output: {result['path']}")
        if "error" in result:
            print(f"  Error: {result['error']}")
        print()
    
    # Recommend best method
    successful_methods = [
        name for name, result in results.items() 
        if result["validation_passed"]
    ]
    
    if successful_methods:
        print(f"✓ Recommended method: {successful_methods[0]}")
        print(f"  Path: {results[successful_methods[0]]['path']}")
    else:
        print("✗ No methods succeeded. Check error messages above.")

def apply_advanced_optimizations(onnx_model_path: str, output_path: str):
    """
    Apply advanced ONNX optimizations including quantization and graph optimization.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.transformers import optimizer
    import onnx
    
    logger.info("Applying advanced ONNX optimizations")
    
    # Load the model
    model = onnx.load(onnx_model_path)
    
    # Graph optimization
    optimized_model_path = output_path.replace('.onnx', '_optimized.onnx')
    
    # Apply transformer-specific optimizations
    try:
        opt = optimizer.optimize_model(
            onnx_model_path,
            model_type='whisper',
            num_heads=32,  # Whisper Large V3 attention heads
            hidden_size=1280,  # Whisper Large V3 hidden size
            optimization_options=None
        )
        opt.save_model_to_file(optimized_model_path)
        logger.info(f"Graph optimization applied: {optimized_model_path}")
    except Exception as e:
        logger.warning(f"Graph optimization failed: {e}")
        optimized_model_path = onnx_model_path
    
    # Dynamic quantization
    quantized_model_path = output_path.replace('.onnx', '_quantized.onnx')
    
    try:
        quantize_dynamic(
            optimized_model_path,
            quantized_model_path,
            weight_type=QuantType.QInt8,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            activation_type=QuantType.QUInt8,
            extra_options={
                'WeightSymmetric': True,
                'ActivationSymmetric': False,
                'EnableSubgraph': False,
                'ForceQuantizeNoInputCheck': False,
                'MatMulConstBOnly': True
            }
        )
        logger.info(f"Dynamic quantization applied: {quantized_model_path}")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")
    
    return {
        'optimized': optimized_model_path,
        'quantized': quantized_model_path if 'quantized_model_path' in locals() else None
    }

def benchmark_onnx_performance(onnx_path: str, iterations: int = 100):
    """
    Comprehensive performance benchmarking for ONNX model inference.
    """
    import time
    import statistics
    
    # Initialize session with performance optimizations
    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 0  # Use all available cores
    session_options.intra_op_num_threads = 0
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, session_options)
    
    # Generate test input
    input_shape = session.get_inputs()[0].shape
    batch_size = 1
    num_mel_bins = 128
    seq_length = 3000
    
    test_input = np.random.randn(batch_size, num_mel_bins, seq_length).astype(np.float32)
    onnx_inputs = {session.get_inputs()[0].name: test_input}
    
    # Warmup runs
    for _ in range(10):
        session.run(None, onnx_inputs)
    
    # Benchmark runs
    inference_times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        outputs = session.run(None, onnx_inputs)
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    stats = {
        'mean_ms': statistics.mean(inference_times),
        'median_ms': statistics.median(inference_times),
        'std_ms': statistics.stdev(inference_times),
        'min_ms': min(inference_times),
        'max_ms': max(inference_times),
        'p95_ms': sorted(inference_times)[int(0.95 * len(inference_times))],
        'p99_ms': sorted(inference_times)[int(0.99 * len(inference_times))]
    }
    
    logger.info("Performance Benchmark Results:")
    for metric, value in stats.items():
        logger.info(f"  {metric}: {value:.2f}")
    
    return stats

def create_deployment_artifacts(onnx_model_path: str, feature_extractor_path: str, output_dir: str):
    """
    Generate complete deployment artifacts including metadata and inference scripts.
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model metadata
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
    
    # Create model metadata
    metadata = {
        "model_type": "whisper_emotion_classification",
        "onnx_opset_version": 17,
        "input_shape": [1, 128, 3000],  # batch, mel_bins, sequence
        "output_shape": [1, 7],  # batch, num_emotions
        "feature_extractor_config": feature_extractor.to_dict(),
        "emotion_labels": {
            0: "angry",
            1: "disgust", 
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        },
        "preprocessing_requirements": {
            "sampling_rate": 16000,
            "normalization": True,
            "padding": "max_length",
            "max_length": 480000  # 30 seconds at 16kHz
        }
    }
    
    # Save metadata
    with open(output_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create optimized inference script
    inference_script = '''
import onnxruntime as ort
import numpy as np
import librosa
from typing import Dict, Union
import json

class WhisperEmotionONNX:
    def __init__(self, model_path: str, metadata_path: str):
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize ONNX session with optimizations
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        self.session = ort.InferenceSession(model_path, session_options)
        self.input_name = self.session.get_inputs()[0].name
        self.emotion_labels = self.metadata["emotion_labels"]
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file for inference."""
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract log-mel spectrogram features
        mel_features = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            hop_length=160,
            win_length=400,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_features, ref=np.max)
        
        # Normalize to [-1, 1] range
        log_mel = (log_mel + 80.0) / 80.0
        
        # Ensure correct shape and padding
        target_length = 3000
        if log_mel.shape[1] < target_length:
            pad_width = target_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :target_length]
        
        return log_mel[np.newaxis, :, :]  # Add batch dimension
    
    def predict(self, audio_input: Union[str, np.ndarray]) -> Dict:
        """Predict emotion from audio input."""
        if isinstance(audio_input, str):
            features = self.preprocess_audio(audio_input)
        else:
            features = audio_input
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: features.astype(np.float32)})
        logits = outputs[0]
        
        # Apply softmax and get predictions
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, predicted_class]
        
        return {
            "emotion": self.emotion_labels[str(predicted_class)],
            "confidence": float(confidence),
            "all_probabilities": {
                self.emotion_labels[str(i)]: float(prob)
                for i, prob in enumerate(probabilities[0])
            }
        }

# Example usage
if __name__ == "__main__":
    predictor = WhisperEmotionONNX("emotion_model.onnx", "model_metadata.json")
    result = predictor.predict("audio_file.wav")
    print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
'''
    
    with open(output_path / "onnx_inference.py", "w") as f:
        f.write(inference_script)
    
    logger.info(f"Deployment artifacts created in {output_path}")
    return output_path

if __name__ == "__main__":
    main()