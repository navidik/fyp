# Convenience imports for the transcription utility package.
__version__ = '1.0.0'
__author__ = 'FYP 6'

# Keep package imports forgiving so optional ML/export dependencies can fail lazily.
try:
    from .transcriber import WhisperTranscriber
    from .diarizer import SpeakerDiarizer
    from .merger import TranscriptMerger
    from .export import TranscriptExporter
except ImportError:
    pass
