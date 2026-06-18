# Whisper transcription helpers used by the Flask routes and WER evaluator.
import logging
import os
from pathlib import Path

import torch
import whisper

logger = logging.getLogger(__name__)

MODEL_NAME = 'base'
SUPPORTED_MODELS = ('tiny', 'base', 'small', 'medium', 'large')

# Default settings used when callers do not override model or device.
WHISPER_CONFIG = {
    'model': MODEL_NAME,
    'device': 'cpu',
    'language': 'en',
    'temperature': 0,
    'fp16': False,
}

SUPPORTED_FORMATS = {
    'wav': 'wav',
    'mp3': 'mp3',
    'm4a': 'm4a',
    'flac': 'flac',
    'ogg': 'ogg'
}


class WhisperTranscriber:
    #Small wrapper that loads one Whisper model and transcribes audio files

    def __init__(self, model_name=MODEL_NAME, device='cpu'):
        #Load the selected Whisper model once so later calls can reuse it
        logger.info(f"Initializing Whisper with model: {model_name}")
        logger.info(f"Device: {device}")

        try:
            self.model = whisper.load_model(
                model_name,
                device=device,
                download_root=None
            )
            self.model_name = model_name
            self.device = device
            logger.info(f"✓ Model loaded: {model_name}")

        except Exception as exc:
            logger.error(f"Failed to load model: {str(exc)}")
            raise Exception(f"Whisper initialization failed: {str(exc)}")

    def transcribe(self, audio_filepath):
        #Validate the file, run Whisper, and return raw text plus timed segments
        logger.info(f"Starting transcription: {audio_filepath}")

        try:
            if not os.path.exists(audio_filepath):
                logger.error(f"File not found: {audio_filepath}")
                return {
                    'success': False,
                    'error': f'Audio file not found: {audio_filepath}',
                    'error_type': 'FILE_NOT_FOUND'
                }

            file_ext = Path(audio_filepath).suffix.lower().lstrip('.')
            if file_ext not in SUPPORTED_FORMATS:
                logger.error(f"Unsupported format: {file_ext}")
                return {
                    'success': False,
                    'error': f'Unsupported audio format: {file_ext}. Supported: {list(SUPPORTED_FORMATS.keys())}',
                    'error_type': 'AUDIO_ERROR'
                }

            audio_data = whisper.load_audio(audio_filepath)
            duration_seconds = len(audio_data) / whisper.audio.SAMPLE_RATE
            logger.info(f"Audio loaded: {len(audio_data)} samples, {whisper.audio.SAMPLE_RATE}Hz")
            logger.info("Running Whisper transcription...")

            whisper_result = self.model.transcribe(
                audio_filepath,
                language='en',
                temperature=0,
                verbose=False,
                fp16=self.device == 'cuda'
            )
            logger.info(f"✓ Transcription complete")

            formatted_result = {
                'success': True,
                'text': whisper_result['text'].strip(),
                'language': whisper_result['language'],
                'segments': whisper_result['segments'],
                'duration': duration_seconds
            }
            logger.info(f"Transcribed {len(whisper_result['segments'])} segments")
            return formatted_result

        except Exception as exc:
            logger.error(f"Transcription error: {str(exc)}")
            return {
                'success': False,
                'error': f'Transcription failed: {str(exc)}',
                'error_type': 'TRANSCRIPTION_ERROR'
            }

    def extract_timestamped_segments(self, audio_filepath):
        #Return a beginner-friendly list of segments with formatted timestamps
        logger.info(f"Extracting timestamped segments from: {audio_filepath}")
        result = self.transcribe(audio_filepath)
        if not result['success']:
            return result

        timestamped_segments = []
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            timestamp = self._seconds_to_timestamp(start_time)
            timestamped_segments.append({
                'timestamp': timestamp,
                'text': segment['text'].strip(),
                'start_seconds': start_time,
                'end_seconds': end_time,
                'duration': end_time - start_time,
                'segment_id': segment['id']
            })
        logger.info(f"Extracted {len(timestamped_segments)} timestamped segments")
        return {
            'success': True,
            'segments': timestamped_segments,
            'total_duration': result['duration']
        }

    @staticmethod
    def _seconds_to_timestamp(seconds):
        #Convert a floating-point second offset into HH:MM:SS
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

_TRANSCRIBER_CACHE = {}


def transcribe_audio_file(filepath, model_name=MODEL_NAME, device='cpu'):
    #Public helper that validates model/device and reuses cached Whisper instances
    if model_name not in SUPPORTED_MODELS:
        logger.warning(f"Unsupported Whisper model requested: {model_name}. Falling back to {MODEL_NAME}.")
        model_name = MODEL_NAME

    requested_device = (device or 'cpu').lower()
    actual_device = requested_device

    if requested_device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        actual_device = 'cpu'

    cache_key = (model_name, actual_device)

    try:
        if cache_key not in _TRANSCRIBER_CACHE:
            _TRANSCRIBER_CACHE[cache_key] = WhisperTranscriber(model_name, device=actual_device)

        result = _TRANSCRIBER_CACHE[cache_key].transcribe(filepath)
        if result.get('success'):
            result['model'] = model_name
            result['device'] = actual_device
            result['requested_device'] = requested_device
        return result
    except Exception as exc:
        logger.error(f"Error: {str(exc)}")
        return {
            'success': False,
            'error': str(exc),
            'error_type': 'INITIALIZATION_ERROR'
        }
