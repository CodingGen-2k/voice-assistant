# TODO- Install Required Libraries+ logging setup
import asyncio
import sounddevice as sd
import numpy as np
import torch
import pyttsx3
import logging
from sounddevice import PortAudioError


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------
# Preload VAD Model 
# ----------------------------
try:
    logging.info("Loading Silero VAD model...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )           
    get_speech_timestamps, _, _, _, _ = vad_utils
    logging.info("VAD model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load VAD model: {e}")
    vad_model, get_speech_timestamps = None, None


# TODO-Record Audio from Microphone
def record_audio(duration=2, sample_rate=16000):
    """
    Record audio from microphone.

    Args:
        duration (float): Recording duration in seconds.
        sample_rate (int): Sampling rate in Hz (default 16kHz).

    Returns:
        numpy.ndarray: Flattened audio array in 16-bit PCM.
    """
    try:
        logging.info(f"Recording audio for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()  # wait until recording is finished
        audio = audio.flatten()
        logging.info(f"Recording complete. Audio length: {len(audio)} samples")
        return audio
    except PortAudioError as e:
        logging.error(f"Microphone access error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error recording audio: {e}")
        return None


# TODO-Voice Activity Detection (VAD)
def vad_detect(audio, sample_rate=16000):
    """
    Detect if there is speech in the audio.

    Args:
        audio (numpy.ndarray): 16-bit PCM audio
        sample_rate (int): Audio sample rate

    Returns:
        bool: True if speech is detected, False otherwise
    """
    if vad_model is None or get_speech_timestamps is None:
        logging.error("VAD model not loaded. Skipping speech detection.")
        return False

    try:
        speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=sample_rate)
        if speech_timestamps:
            logging.info("Speech detected.")
            return True
        else:
            logging.info("No speech detected.")
            return False
    except Exception as e:
        logging.error(f"Error during VAD detection: {e}")
        return False


# TODO-Combine Audio + VAD + Prompt

engine = pyttsx3.init()

async def speak(message: str):
    """Speak out a message asynchronously (non-blocking in async flow)."""
    logging.info(f"TTS Prompt: {message}")
    try:
        engine.say(message)
        engine.runAndWait()  # blocking but okay for small prompts
    except Exception as e:
        logging.error(f"TTS error: {e}")
    await asyncio.sleep(0.1)


# async listen function
async def listen_for_speech(prompt_message="Please speak now...", 
                             sample_rate=16000, 
                             chunk_duration=2, 
                             silence_limit=5,   # stop after 3 silent chunks in a row
                             max_duration=30):
    """
    Continuously listen to the user until speech is detected.
    
    Args:
        prompt_message (str): TTS prompt if no speech detected
        sample_rate (int): Audio sample rate
        chunk_duration (float): Duration per audio chunk

    Returns:
        numpy.ndarray: Audio chunk where speech was detected
    """
    collected_audio = []
    silence_counter = 0
    total_time = 0
    speech_started = False

    while True:
        # Record one chunk
        audio = record_audio(duration=chunk_duration, sample_rate=sample_rate)
        if audio is None:
            await speak("Microphone error. Please check your device.")
            break

        total_time += chunk_duration
        is_speech = vad_detect(audio, sample_rate=sample_rate)

        if is_speech:
            speech_started = True
            silence_counter = 0
            collected_audio.append(audio)
            logging.info("Speech detected, appending chunk.")
        else:
            if speech_started:  # speech had started before
                silence_counter += 1
                logging.info(f"Silence detected ({silence_counter}/{silence_limit})")
                if silence_counter >= silence_limit:
                    logging.info("User finished speaking.")
                    break
            else:
                # still waiting for first speech
                await speak(prompt_message)

        # Timeout check
        if total_time >= max_duration:
            logging.info("Timeout reached. Stopping.")
            break

    if collected_audio:
        final_audio = np.concatenate(collected_audio)
        return final_audio
    else:
        return None

# Example usage
if __name__ == "__main__":
    async def main():
        audio_data = await listen_for_speech()
        if audio_data is not None:
            logging.info(f"Audio received. Length: {len(audio_data)} samples")
        else:
            logging.warning("No audio collected.")
        
    asyncio.run(main())