# WER/CER evaluation helpers for comparing Whisper output to reference text.
import logging
import re
from pathlib import Path

from utils.transcriber import transcribe_audio_file

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation_results"
WER_OUTPUT_DIR = EVALUATION_ROOT / "wer_outputs"
WER_REPORT_PATH = EVALUATION_ROOT / "wer_report.csv"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


def _load_dependencies():
    # Import optional evaluation libraries only when an evaluation route needs them.
    try:
        import pandas as pd
        from jiwer import cer, wer
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependencies. Install: pip install jiwer datasets soundfile pyannote.metrics pyannote.core pandas"
        ) from exc

    return pd, wer, cer


def ensure_directories():
    # Create the folders that store generated WER transcripts and reports.
    WER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    # Lowercase and remove punctuation so WER/CER compare words consistently.
    if text is None:
        return ""

    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _read_reference_text(reference_text_path):
    # Load and validate the human-written reference transcript.
    reference_path = Path(reference_text_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference transcript file not found: {reference_text_path}")

    reference_text = reference_path.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError("Reference transcript is empty.")

    return reference_text


def _save_system_transcript(audio_path, system_text):
    # Persist Whisper's generated transcript for later inspection/download.
    ensure_directories()
    output_name = f"{Path(audio_path).stem}_system_transcript.txt"
    output_path = WER_OUTPUT_DIR / output_name
    output_path.write_text(system_text, encoding="utf-8")
    return str(output_path)


def _calculate_transcription_metrics(reference_text, system_text):
    # Calculate WER, CER, and a simple 100-WER accuracy score.
    _, wer_fn, cer_fn = _load_dependencies()

    normalized_reference = normalize_text(reference_text)
    normalized_system = normalize_text(system_text)

    if not normalized_reference:
        raise ValueError("Normalized reference transcript is empty.")

    if not normalized_system:
        raise ValueError("Normalized system transcript is empty.")

    wer_percentage = round(wer_fn(normalized_reference, normalized_system) * 100, 2)
    cer_percentage = round(cer_fn(normalized_reference, normalized_system) * 100, 2)
    transcription_accuracy = round(max(0.0, 100 - wer_percentage), 2)

    return wer_percentage, cer_percentage, transcription_accuracy


def evaluate_wer(audio_path, reference_text_path):
    # Transcribe one audio file and compare it with one reference transcript.
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError("Missing audio file for WER evaluation.")

    if not reference_text_path or not Path(reference_text_path).exists():
        raise FileNotFoundError("Missing reference transcript text file for WER evaluation.")

    reference_text = _read_reference_text(reference_text_path)

    transcription_result = transcribe_audio_file(str(audio_path))
    if not transcription_result.get("success"):
        error_message = transcription_result.get("error", "Unknown transcription failure.")
        raise RuntimeError(f"Transcription failure: {error_message}")

    system_text = (transcription_result.get("text") or "").strip()
    if not system_text:
        raise ValueError("System transcript is empty.")

    wer_percentage, cer_percentage, transcription_accuracy = _calculate_transcription_metrics(
        reference_text,
        system_text,
    )

    output_transcript_path = _save_system_transcript(audio_path, system_text)

    return {
        "wer": wer_percentage,
        "cer": cer_percentage,
        "transcription_accuracy": transcription_accuracy,
        "reference_text": reference_text,
        "system_text": system_text,
        "output_transcript_path": output_transcript_path,
    }


def evaluate_wer_folder(audio_folder_path, transcript_folder_path, progress_callback=None, report_csv_path=None):
    # Evaluate matching audio/transcript files across a folder and write a CSV report.
    pd, _, _ = _load_dependencies()
    ensure_directories()

    audio_folder = Path(audio_folder_path)
    transcript_folder = Path(transcript_folder_path)

    if not audio_folder.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_folder_path}")

    if not transcript_folder.exists():
        raise FileNotFoundError(f"Transcript folder not found: {transcript_folder_path}")

    audio_files = sorted(
        file_path
        for file_path in audio_folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    # Files are paired by basename, for example meeting01.wav with meeting01.txt.
    transcript_map = {
        file_path.stem: file_path
        for file_path in transcript_folder.rglob("*.txt")
        if file_path.is_file()
    }

    file_results = []
    errors = []
    report_rows = []

    total_files = len(audio_files)

    for index, audio_file in enumerate(audio_files, start=1):
        if progress_callback:
            progress_callback(
                {
                    "stage": "processing",
                    "current_file": audio_file.name,
                    "processed_files": index - 1,
                    "total_files": total_files,
                    "percent": round(((index - 1) / total_files) * 100, 2) if total_files else 0.0,
                    "message": f"Transcribing and evaluating {audio_file.name} ({index}/{total_files})",
                }
            )

        reference_path = transcript_map.get(audio_file.stem)
        if reference_path is None:
            error_message = f"Matching transcript not found for {audio_file.name}"
            errors.append({"file_name": audio_file.name, "error": error_message})
            report_rows.append(
                {
                    "file_name": audio_file.name,
                    "wer": None,
                    "cer": None,
                    "transcription_accuracy": None,
                    "status": "failed",
                    "error": error_message,
                }
            )
            if progress_callback:
                progress_callback(
                    {
                        "stage": "processing",
                        "current_file": audio_file.name,
                        "processed_files": index,
                        "total_files": total_files,
                        "percent": round((index / total_files) * 100, 2) if total_files else 100.0,
                        "message": f"Skipped {audio_file.name}: matching transcript not found.",
                    }
                )
            continue

        try:
            result = evaluate_wer(str(audio_file), str(reference_path))
            file_result = {
                "file_name": audio_file.name,
                "wer": result["wer"],
                "cer": result["cer"],
                "transcription_accuracy": result["transcription_accuracy"],
                "status": "success",
            }
            file_results.append(file_result)
            report_rows.append({**file_result, "error": ""})
        except Exception as exc:
            error_message = str(exc)
            logger.exception("Bulk WER evaluation failed for %s", audio_file)
            errors.append({"file_name": audio_file.name, "error": error_message})
            report_rows.append(
                {
                    "file_name": audio_file.name,
                    "wer": None,
                    "cer": None,
                    "transcription_accuracy": None,
                    "status": "failed",
                    "error": error_message,
                }
            )
        finally:
            if progress_callback:
                progress_callback(
                    {
                        "stage": "processing",
                        "current_file": audio_file.name,
                        "processed_files": index,
                        "total_files": total_files,
                        "percent": round((index / total_files) * 100, 2) if total_files else 100.0,
                        "message": f"Completed {audio_file.name} ({index}/{total_files})",
                    }
                )

    successful_files = len(file_results)
    failed_files = len(errors)

    if successful_files > 0:
        average_wer = round(sum(item["wer"] for item in file_results) / successful_files, 2)
        average_cer = round(sum(item["cer"] for item in file_results) / successful_files, 2)
        average_transcription_accuracy = round(
            sum(item["transcription_accuracy"] for item in file_results) / successful_files,
            2,
        )
    else:
        average_wer = 0.0
        average_cer = 0.0
        average_transcription_accuracy = 0.0

    output_report_path = Path(report_csv_path) if report_csv_path else WER_REPORT_PATH
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(output_report_path, index=False)

    result = {
        "total_files": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "average_wer": average_wer,
        "average_cer": average_cer,
        "average_transcription_accuracy": average_transcription_accuracy,
        "report_csv_path": str(output_report_path),
        "file_results": file_results,
        "errors": errors,
    }

    if progress_callback:
        progress_callback(
            {
                "stage": "completed",
                "current_file": None,
                "processed_files": total_files,
                "total_files": total_files,
                "percent": 100.0,
                "message": "Bulk WER evaluation completed.",
                "result": result,
            }
        )

    return result
