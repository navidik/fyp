# Pyannote speaker diarization helpers for assigning speakers to audio regions.
import os
import logging
import inspect
import re

import huggingface_hub
import torch
import torchaudio
from dotenv import load_dotenv


def _patch_hf_hub_download_for_pyannote():
    # Bridge pyannote's older `use_auth_token` call to newer huggingface_hub.
    signature = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" in signature.parameters:
        return

    original_download = huggingface_hub.hf_hub_download

    def compatible_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = compatible_hf_hub_download


_patch_hf_hub_download_for_pyannote()

from pyannote.audio import Pipeline

load_dotenv()

logger = logging.getLogger(__name__)

DIARIZATION_CONFIG = {
    'model': 'pyannote/speaker-diarization-3.0',
    'device': 'cpu',
    'num_speakers': None,
}

DIARIZATION_PARAMS = {
    'segmentation': {
        'threshold': 0.5,
    },
    'clustering': {
        'threshold': 7.5,
    }
}


def _load_pyannote_pipeline(model_name, hf_token=None):
    # Load a Pyannote pipeline using whichever token argument this version supports.
    signature = inspect.signature(Pipeline.from_pretrained)
    kwargs = {}

    if hf_token:
        if "token" in signature.parameters:
            kwargs["token"] = hf_token
        elif "use_auth_token" in signature.parameters:
            kwargs["use_auth_token"] = hf_token

    return Pipeline.from_pretrained(model_name, **kwargs)


def _get_annotation_from_diarization(diarization):
    # Normalize different Pyannote return shapes to an Annotation-like object.
    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization

    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization

    return diarization


class SpeakerDiarizer:
    # Loads a Pyannote pipeline and extracts speaker time ranges from audio.
    def __init__(self, model_name='pyannote/speaker-diarization-3.0', device='cpu'):
        # Initialize Pyannote, including Hugging Face token support when configured.
        logger.info(f"Initializing Diarization with model: {model_name}")
        logger.info(f"Device: {device}")

        try:
            logger.info("Loading Pyannote pipeline...")
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            self.pipeline = _load_pyannote_pipeline(model_name, hf_token)
            self.pipeline.to(torch.device(device))
            self.model_name = model_name
            self.device = device
            logger.info(f"✓ Diarization pipeline loaded successfully")

        except Exception as exc:
            error_msg = str(exc)
            if 'Accept the license' in error_msg or 'You must accept' in error_msg:
                logger.error("❌ License not accepted")
                logger.error("Please follow these steps:")
                logger.error("1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.0")
                logger.error("2. Click 'Accept repository'")
                logger.error("3. Visit: https://huggingface.co/pyannote/segmentation-3.0")
                logger.error("4. Click 'Accept repository'")
                logger.error("5. Login: python -c 'from huggingface_hub import hf_login; hf_login()'")
            logger.error(f"Initialization failed: {error_msg}")
            raise Exception(f"Diarization initialization failed: {error_msg}")

    def diarize(self, audio_filepath):
        # Run the diarization pipeline and return speaker segments for merging.
        logger.info(f"Starting diarization: {audio_filepath}")

        try:
            if not os.path.exists(audio_filepath):
                logger.error(f"File not found: {audio_filepath}")
                return {
                    'success': False,
                    'error': f'Audio file not found: {audio_filepath}',
                    'error_type': 'FILE_NOT_FOUND'
                }

            logger.info("Loading audio file...")
            waveform, sample_rate = torchaudio.load(audio_filepath)
            logger.info(f"Audio loaded: channels={waveform.shape[0]}, sr={sample_rate}Hz")

            if waveform.shape[0] > 1:
                logger.info("Converting stereo to mono...")
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pyannote accepts an in-memory waveform dictionary, avoiding another file read.
            audio = {
                'waveform': waveform,
                'sample_rate': sample_rate,
                'duration': waveform.shape[1] / sample_rate
            }
            logger.info(f"Audio duration: {audio['duration']:.1f} seconds")
            logger.info("Running diarization pipeline...")
            diarization = self.pipeline(audio)
            logger.info("✓ Diarization complete")
            segments = self._extract_segments(diarization, audio['duration'])
            num_speakers = len({seg['speaker_id'] for seg in segments})
            result = {
                'success': True,
                'segments': segments,
                'num_speakers': num_speakers,
                'duration': audio['duration'],
                'diarization_object': diarization
            }
            logger.info(f"Diarization result: {num_speakers} speakers, {len(segments)} segments")
            return result

        except Exception as exc:
            logger.error(f"Diarization error: {str(exc)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Diarization failed: {str(exc)}',
                'error_type': 'DIARIZATION_ERROR'
            }

    def _extract_segments(self, diarization, _total_duration):
        # Convert Pyannote tracks into simple dictionaries with stable speaker labels.
        annotation = _get_annotation_from_diarization(diarization)
        segments = []
        speaker_ids = {}

        for segment, track, speaker in annotation.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time

            speaker_label = str(speaker)
            speaker_number = re.search(r"(\d+)$", speaker_label)
            if speaker_number:
                # Pyannote labels such as SPEAKER_00 become Speaker 1 in the UI.
                speaker_id = int(speaker_number.group(1))
            else:
                if speaker_label not in speaker_ids:
                    speaker_ids[speaker_label] = len(speaker_ids)
                speaker_id = speaker_ids[speaker_label]

            start_timestamp = self._seconds_to_timestamp(start_time)
            end_timestamp = self._seconds_to_timestamp(end_time)
            segment_dict = {
                'speaker': f'Speaker {speaker_id + 1}',
                'speaker_id': speaker_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'track': track
            }
            segments.append(segment_dict)
        return segments

    @staticmethod
    def _seconds_to_timestamp(seconds):
        # Convert seconds to HH:MM:SS for display and export.
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_DIARIZER_CACHE = {}


def diarize_audio_file(filepath, model_name='pyannote/speaker-diarization-3.0', device='cpu'):
    # Public helper that validates device choice and reuses cached Pyannote pipelines.
    requested_device = (device or 'cpu').lower()
    actual_device = requested_device

    if requested_device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested for diarization but not available. Falling back to CPU.")
        actual_device = 'cpu'

    cache_key = (model_name, actual_device)

    try:
        if cache_key not in _DIARIZER_CACHE:
            _DIARIZER_CACHE[cache_key] = SpeakerDiarizer(model_name=model_name, device=actual_device)

        result = _DIARIZER_CACHE[cache_key].diarize(filepath)
        if result.get('success'):
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
