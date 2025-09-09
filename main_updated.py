
import asyncio
import sounddevice as sd
import numpy as np
import torch
import pyttsx3
import logging
import atexit
from sounddevice import PortAudioError
from faster_whisper import WhisperModel

# ================== CONFIG ==================
SAMPLE_RATE = 16000
CHUNK_DURATION = 1       # seconds per audio chunk
SILENCE_LIMIT = 5        # chunks after speech ends
MAX_DURATION = 30        # seconds max listen
VAD_THRESHOLD = 0.5     # lower = more sensitive
WHISPER_MODEL_SIZE = "small" 
DEVICE = "cuda"          # "cuda" or "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Silero VAD
try:
    logging.info("Loading Silero VAD model...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )
    get_speech_timestamps, _, _, _, _ = vad_utils
    logging.info("Silero VAD model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Silero VAD model: {e}")
    vad_model, get_speech_timestamps = None, None

# Load Faster-Whisper ASR
try:
    logging.info(f"Loading Faster-Whisper model: {WHISPER_MODEL_SIZE} on {DEVICE}...")
    asr_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logging.info("Faster-Whisper ASR initialized successfully.")
except Exception as e:
    logging.error(f"Failed to load Faster-Whisper: {e}")
    asr_model = None

# Initialize TTS engine
try:
    engine = pyttsx3.init()
    logging.info("TTS engine initialized successfully.")
    atexit.register(engine.stop)  # ensure clean shutdown
except Exception as e:
    logging.error(f"Failed to initialize TTS engine: {e}")
    engine = None


# ------------------- UTILITIES -------------------

def record_audio(duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE) -> np.ndarray | None:
    """Record audio safely with logging and error handling."""
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        return audio.flatten()
    except PortAudioError as e:
        logging.error(f"Microphone error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during audio recording: {e}")
    return None


def streaming_vad(audio: np.ndarray, sample_rate=SAMPLE_RATE) -> bool:
    """Run Silero VAD on an audio chunk."""
    if vad_model is None or get_speech_timestamps is None:
        logging.error("VAD model not loaded. Skipping detection.")
        return False

    try:
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_model,
            sampling_rate=sample_rate,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=150,
            min_silence_duration_ms=100
        )
        detected = bool(speech_timestamps)
        logging.debug(f"VAD detected speech: {detected}")
        return detected
    except Exception as e:
        logging.error(f"Error during VAD detection: {e}")
        return False


async def speak(message: str):
    """Speak text out loud using pyttsx3."""
    if engine is None:
        logging.error("TTS engine not initialized.")
        return
    try:
        logging.info(f"TTS: {message}")
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")
    await asyncio.sleep(0.05)


def transcribe_audio(audio: np.ndarray, sample_rate=SAMPLE_RATE) -> str | None:
    """Transcribe audio using Faster-Whisper ASR."""
    if asr_model is None:
        logging.error("ASR model not initialized.")
        return None

    try:
        logging.info("Starting ASR transcription...")
        audio_float32 = audio.astype(np.float32) / 32768.0  # convert int16 → float32 [-1,1]
        segments, _ = asr_model.transcribe(audio_float32, language="en")

        transcription = " ".join([seg.text for seg in segments])
        logging.info(f"ASR Result: {transcription}")
        return transcription
    except Exception as e:
        logging.error(f"ASR transcription failed: {e}")
        return None


# ------------------- CORE LISTEN FUNCTION -------------------

async def listen_for_speech(
    prompt_message="Please speak now...",
    sample_rate=SAMPLE_RATE,
    chunk_duration=CHUNK_DURATION,
    silence_limit=SILENCE_LIMIT,
    max_duration=MAX_DURATION
) -> np.ndarray | None:
    """
    Continuously listen to the microphone with VAD.
    Collects audio until user finishes speaking or max duration is reached.
    """
    collected_audio = []
    silence_counter = 0
    total_time = 0
    speech_started = False

    logging.info("Listening for speech...")

    while True:
        audio = record_audio(duration=chunk_duration, sample_rate=sample_rate)
        if audio is None:
            await speak("Microphone error. Please check your device.")
            break

        total_time += chunk_duration
        is_speech = streaming_vad(audio, sample_rate=sample_rate)

        if is_speech:
            speech_started = True
            silence_counter = 0
            collected_audio.append(audio)
            logging.info("Speech detected, appending chunk.")
        else:
            if speech_started:
                silence_counter += 1
                logging.info(f"Silence detected ({silence_counter}/{silence_limit})")
                if silence_counter >= silence_limit:
                    logging.info("User finished speaking.")
                    break
            else:
                await speak(prompt_message)

        if total_time >= max_duration:
            logging.warning("Max listening duration reached.")
            break

    if collected_audio:
        final_audio = np.concatenate(collected_audio)
        logging.info(f"Final audio collected: {len(final_audio)} samples.")
        return final_audio

    logging.warning("No speech collected.")
    return None


# ------------------- ENTRYPOINT -------------------

if __name__ == "__main__":
    async def main():
        try:
            audio_data = await listen_for_speech()
            if audio_data is not None:
                transcription = transcribe_audio(audio_data)
                if transcription:
                    await speak(f"You said: {transcription}")
            else:
                logging.warning("No usable audio captured.")
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting gracefully.")
        except Exception as e:
            logging.error(f"Fatal error in main loop: {e}")

    asyncio.run(main())