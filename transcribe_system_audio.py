"""Real-time system audio transcription using Whisper."""
import os
import numpy as np
import soundcard as sc
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv

from utils.device import get_optimal_device
from utils.logger_config import setup_logger

# Initialize logging
logger = setup_logger()

def load_model():
    """Load and prepare the Whisper model."""
    load_dotenv()
    model_id = os.getenv('MODEL_ID', 'openai/whisper-base')
    device, dtype = get_optimal_device()
    
    logger.info(f"Loading model {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    
    if dtype == torch.float16 and device.type == "cuda":
        model = model.half()
    
    return processor, model, device, dtype

def get_loopback_device():
    """Get the default system audio loopback device."""
    try:
        # List all available microphones
        mics = sc.all_microphones()
        logger.info("Available recording devices:")
        for mic in mics:
            logger.info(f"- {mic.name}")
        
        # Look for Voicemeeter device
        for mic in mics:
            if "voicemeeter" in mic.name.lower():
                return mic
        
        # If no Voicemeeter device found, try to get default input device
        default_mic = sc.default_microphone()
        if default_mic:
            return default_mic
        
        raise RuntimeError("No suitable audio device found")
    except Exception as e:
        logger.error(f"Error finding audio device: {e}")
        raise

def process_audio(audio_data, processor, model, device):
    """Process audio chunk through Whisper model."""
    try:
        # Convert audio to float32 if not already
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Prepare audio features
        input_features = processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return ""

def main():
    """Main execution function."""
    try:
        # Load environment variables
        load_dotenv()
        chunk_duration = float(os.getenv('CHUNK_DURATION_S', '3'))
        silence_threshold = float(os.getenv('SILENCE_THRESHOLD_RMS', '0.005'))
        
        # Initialize model and audio device
        processor, model, device, _ = load_model()
        loopback_device = get_loopback_device()
        
        logger.info(f"Recording from: {loopback_device.name}")
        logger.info("Started listening... (Press Ctrl+C to stop)")
        
        # Start recording loop
        with loopback_device.recorder(samplerate=16000) as mic:
            while True:
                # Record audio chunk
                audio_data = mic.record(int(16000 * chunk_duration))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Check if audio is not silence
                if np.sqrt(np.mean(audio_data**2)) > silence_threshold:
                    transcription = process_audio(audio_data, processor, model, device)
                    if transcription:
                        print(f"🎤 {transcription}")
    
    except KeyboardInterrupt:
        logger.info("\nStopping transcription...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()