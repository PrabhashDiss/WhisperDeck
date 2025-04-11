"""Real-time system audio transcription using Whisper."""
import os
import numpy as np
import soundcard as sc
import torch
import warnings
from datetime import datetime
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dotenv import load_dotenv
from collections import deque

from utils.device import get_optimal_device
from utils.logger_config import setup_logger

# Filter out soundcard data discontinuity warnings
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# Initialize logging
logger = setup_logger()

# Constants for audio processing
SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 30  # Minimum seconds of audio needed for Whisper
OVERLAP_DURATION = 0.5  # 500ms overlap between chunks
AUDIO_BUFFER_DURATION = max(MIN_AUDIO_LENGTH, 30.0)  # Keep enough audio history for minimum length

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
        
        # Ensure minimum length requirement
        min_samples = SAMPLE_RATE * MIN_AUDIO_LENGTH
        if len(audio_data) < min_samples:
            # Pad with silence if needed
            padding = np.zeros(min_samples - len(audio_data), dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])
        
        # Prepare audio features
        inputs = processor(
            audio_data, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt",
            padding=True
        )
        input_features = inputs.input_features.to(device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=2,
                temperature=0.2
            )
        
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return ""

def get_output_file():
    """Create a timestamped output file in the output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"transcript_{timestamp}.txt"

def main():
    """Main execution function."""
    try:
        # Load environment variables
        load_dotenv()
        chunk_duration = float(os.getenv('CHUNK_DURATION_S', '5'))  # Increased default chunk size
        silence_threshold = float(os.getenv('SILENCE_THRESHOLD_RMS', '0.002'))
        
        # Initialize model and audio device
        processor, model, device, _ = load_model()
        loopback_device = get_loopback_device()
        
        # Create output file
        output_file = get_output_file()
        logger.info(f"Saving transcriptions to: {output_file}")
        
        logger.info(f"Recording from: {loopback_device.name}")
        logger.info("Started listening... (Press Ctrl+C to stop)")
        
        # Calculate sizes in samples
        overlap_samples = int(OVERLAP_DURATION * SAMPLE_RATE)
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        buffer_samples = int(AUDIO_BUFFER_DURATION * SAMPLE_RATE)
        
        # Initialize audio buffer with enough capacity for minimum length requirement
        audio_buffer = deque(maxlen=buffer_samples)
        previous_chunk = np.zeros(overlap_samples, dtype=np.float32)
        
        # Start recording loop
        with loopback_device.recorder(samplerate=SAMPLE_RATE) as mic, \
             open(output_file, 'a', encoding='utf-8') as f:
            while True:
                # Record audio chunk
                audio_data = mic.record(chunk_samples)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Add overlap from previous chunk
                audio_with_overlap = np.concatenate([previous_chunk, audio_data])
                
                # Update previous chunk for next iteration
                previous_chunk = audio_data[-overlap_samples:]
                
                # Extend audio buffer
                audio_buffer.extend(audio_data)
                
                # Check if audio is not silence and we have enough data
                if len(audio_buffer) >= MIN_AUDIO_LENGTH * SAMPLE_RATE and \
                   np.sqrt(np.mean(audio_data**2)) > silence_threshold:
                    # Process audio with context from buffer
                    context_audio = np.array(list(audio_buffer))
                    transcription = process_audio(context_audio, processor, model, device)
                    
                    if transcription:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] {transcription}\n"
                        print(f"🎤 {transcription}")
                        f.write(output_line)
                        f.flush()
    
    except KeyboardInterrupt:
        logger.info("\nStopping transcription...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()