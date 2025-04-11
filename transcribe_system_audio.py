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

from utils.device import get_optimal_device
from utils.logger_config import setup_logger

# Filter out soundcard data discontinuity warnings
warnings.filterwarnings("ignore", message="data discontinuity in recording")

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
        
        # Calculate target length (30 seconds at 16kHz = 480000 samples)
        target_length = 480000
        current_length = len(audio_data)
        
        if current_length < target_length:
            # Pad with zeros if audio is shorter than 30 seconds
            padding = np.zeros(target_length - current_length, dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])
        elif current_length > target_length:
            # Trim if audio is longer than 30 seconds
            audio_data = audio_data[:target_length]
        
        # Prepare audio features with attention mask
        inputs = processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        input_features = inputs.input_features.to(device)
        
        # Create attention mask (all 1s since we're processing a single chunk)
        attention_mask = torch.ones_like(input_features, dtype=torch.long, device=device)
        
        # Generate transcription with attention mask
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask
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
        chunk_duration = float(os.getenv('CHUNK_DURATION_S', '3'))
        silence_threshold = float(os.getenv('SILENCE_THRESHOLD_RMS', '0.005'))
        
        # Initialize model and audio device
        processor, model, device, _ = load_model()
        loopback_device = get_loopback_device()
        
        # Create output file
        output_file = get_output_file()
        logger.info(f"Saving transcriptions to: {output_file}")
        
        logger.info(f"Recording from: {loopback_device.name}")
        logger.info("Started listening... (Press Ctrl+C to stop)")
        
        # Start recording loop
        with loopback_device.recorder(samplerate=16000) as mic, \
             open(output_file, 'a', encoding='utf-8') as f:
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
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        output_line = f"[{timestamp}] {transcription}\n"
                        print(f"🎤 {transcription}")
                        f.write(output_line)
                        f.flush()  # Ensure immediate write to file
    
    except KeyboardInterrupt:
        logger.info("\nStopping transcription...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()