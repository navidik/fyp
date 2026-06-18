# DER evaluation helpers for benchmarking diarization on CALLHOME samples.
import logging
from pathlib import Path

from utils.diarizer import diarize_audio_file

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = PROJECT_ROOT / "uploads" / "callhome"
EVALUATION_ROOT = PROJECT_ROOT / "evaluation_results"
DER_REPORT_PATH = EVALUATION_ROOT / "der_report.csv"
CALLHOME_CONFIG = "eng"


def _load_dependencies():
    # Import optional DER dependencies only when the benchmark is requested.
    try:
        import pandas as pd
        import soundfile as sf
        from datasets import load_dataset
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError as exc:
        raise RuntimeError(
            "Missing evaluation dependencies. Install: pip install jiwer datasets soundfile pyannote.metrics pyannote.core pandas"
        ) from exc

    return pd, sf, load_dataset, Annotation, Segment, DiarizationErrorRate


def ensure_directories():
    # Create folders for temporary CALLHOME audio and DER CSV reports.
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)


def create_reference_annotation(sample, uri):
    # Convert CALLHOME ground-truth speaker timestamps into a Pyannote Annotation.
    _, _, _, Annotation, Segment, _ = _load_dependencies()
    annotation = Annotation(uri=uri)

    starts = sample.get("timestamps_start") or []
    ends = sample.get("timestamps_end") or []
    speakers = sample.get("speakers") or []

    for start, end, speaker in zip(starts, ends, speakers):
        annotation[Segment(float(start), float(end))] = str(speaker)

    return annotation


def create_system_annotation(system_output, uri):
    # Convert this app's diarization output into a Pyannote Annotation.
    _, _, _, Annotation, Segment, _ = _load_dependencies()

    if hasattr(system_output, "itertracks"):
        return system_output

    annotation = Annotation(uri=uri)
    if not isinstance(system_output, list):
        raise ValueError("Unsupported diarization output format.")

    for item in system_output:
        start = item.get("start")
        if start is None:
            start = item.get("start_time")

        end = item.get("end")
        if end is None:
            end = item.get("end_time")

        speaker = item.get("speaker") or item.get("label") or item.get("speaker_id")
        if start is None or end is None or speaker is None:
            continue

        annotation[Segment(float(start), float(end))] = str(speaker)

    return annotation


def save_callhome_audio(sample, output_path):
    # Write one CALLHOME dataset audio sample to disk for Pyannote processing.
    _, sf, _, _, _, _ = _load_dependencies()
    audio = sample["audio"]
    sf.write(output_path, audio["array"], audio["sampling_rate"])


def _load_callhome_samples():
    # Load the CALLHOME dataset split and surface access errors clearly.
    _, _, load_dataset, _, _, _ = _load_dependencies()

    try:
        dataset = load_dataset("talkbank/callhome", CALLHOME_CONFIG)
    except Exception as exc:
        error_message = str(exc)
        if any(token in error_message.lower() for token in ("401", "403", "gated", "access", "login", "auth")):
            raise PermissionError("CALLHOME dataset requires Hugging Face login or access approval.") from exc
        raise

    if "data" in dataset:
        return dataset["data"]
    if "train" in dataset:
        return dataset["train"]
    return next(iter(dataset.values()))


def evaluate_der(num_samples=3):
    # Run diarization on a limited CALLHOME sample set and calculate DER.
    pd, _, _, _, _, DiarizationErrorRate = _load_dependencies()
    ensure_directories()

    dataset = _load_callhome_samples()
    requested_samples = max(1, int(num_samples))
    sample_results = []
    errors = []
    report_rows = []

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

    for index, sample in enumerate(dataset.select(range(min(requested_samples, len(dataset))))):
        sample_id = sample.get("audio", {}).get("path") or f"callhome_{index + 1:03d}"
        sample_uri = Path(sample_id).stem
        output_audio_path = UPLOAD_ROOT / f"{sample_uri}.wav"

        try:
            save_callhome_audio(sample, str(output_audio_path))
            reference = create_reference_annotation(sample, sample_uri)

            diarization_result = diarize_audio_file(str(output_audio_path))
            if not diarization_result.get("success"):
                raise RuntimeError(diarization_result.get("error", "Diarization failed."))

            system_source = diarization_result.get("diarization_object") or diarization_result.get("segments")
            system_annotation = create_system_annotation(system_source, sample_uri)
            der_value = round(metric(reference, system_annotation) * 100, 2)

            duration = round(len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"], 2)
            num_speakers = len(set(sample.get("speakers") or []))
            result_row = {
                "sample_id": sample_uri,
                "duration": duration,
                "num_speakers": num_speakers,
                "der": der_value,
                "status": "success",
            }
            sample_results.append(result_row)
            report_rows.append({**result_row, "error": ""})
        except Exception as exc:
            error_message = str(exc)
            logger.exception("DER evaluation failed for sample %s", sample_uri)
            errors.append({"sample_id": sample_uri, "error": error_message})
            report_rows.append(
                {
                    "sample_id": sample_uri,
                    "duration": None,
                    "num_speakers": None,
                    "der": None,
                    "status": "failed",
                    "error": error_message,
                }
            )

    if sample_results:
        average_der = round(sum(item["der"] for item in sample_results) / len(sample_results), 2)
    else:
        average_der = 0.0

    diarization_accuracy = round(max(0.0, 100 - average_der), 2)

    pd.DataFrame(report_rows).to_csv(DER_REPORT_PATH, index=False)

    return {
        "average_der": average_der,
        "diarization_accuracy": diarization_accuracy,
        "report_csv_path": str(DER_REPORT_PATH),
        "sample_results": sample_results,
        "errors": errors,
    }
