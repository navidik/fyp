# Flask entry point for upload, transcription, diarization, export, and evaluation.
import io
import logging
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from utils.der_evaluator import evaluate_der
from utils.diarizer import diarize_audio_file
from utils.export import export_transcript
from utils.merger import merge_results
from utils.transcriber import MODEL_NAME, SUPPORTED_MODELS, transcribe_audio_file
from utils.wer_evaluator import evaluate_wer, evaluate_wer_folder


load_dotenv()
app = Flask(__name__)

# Core Flask limits and project-owned storage folders.
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['EVALUATION_FOLDER'] = os.path.join(os.path.dirname(__file__), 'evaluation_results')

# Main transcription supports OGG; WER evaluation keeps the original narrower list.
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
WER_ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory status registry for the asynchronous bulk WER route.
BULK_WER_TASKS = {}
BULK_WER_TASKS_LOCK = threading.Lock()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['EVALUATION_FOLDER'], exist_ok=True)

# Evaluation uploads are separated from normal meeting uploads to keep reports tidy.
WER_AUDIO_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'wer_audio')
WER_REFERENCE_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'wer_references')
CALLHOME_FOLDER = os.path.join(app.config['UPLOAD_FOLDER'], 'callhome')
WER_OUTPUT_FOLDER = os.path.join(app.config['EVALUATION_FOLDER'], 'wer_outputs')
DER_OUTPUT_FOLDER = os.path.join(app.config['EVALUATION_FOLDER'], 'der_outputs')

for folder in (WER_AUDIO_FOLDER, WER_REFERENCE_FOLDER, CALLHOME_FOLDER, WER_OUTPUT_FOLDER, DER_OUTPUT_FOLDER):
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    # Return True when an uploaded meeting audio filename has a supported extension.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_id():
    # Create a UUID string used for upload sessions, batches, and background tasks.
    return str(uuid.uuid4())


def get_file_info(filepath):
    # Collect file metadata shown to the frontend after a successful upload.
    if not os.path.exists(filepath):
        return {'exists': False, 'error': 'File not found'}

    stat = os.stat(filepath)
    size_bytes = stat.st_size
    size_mb = size_bytes / (1024 * 1024)
    modified_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

    return {
        'exists': True,
        'size_bytes': size_bytes,
        'size_mb': round(size_mb, 2),
        'size_readable': f'{round(size_mb, 2)} MB',
        'modified': modified_time
    }


def _prepare_accuracy_context():
    # Build the default template context for the accuracy page.
    return {
        'single_wer_result': None,
        'bulk_wer_result': None,
        'der_result': None,
        'page_errors': []
    }


def _create_bulk_wer_task(total_files=0):
    # Register a new background WER task before its worker thread starts.
    task_id = generate_unique_id()
    task = {
        'task_id': task_id,
        'status': 'queued',
        'percent': 0.0,
        'processed_files': 0,
        'total_files': total_files,
        'current_file': None,
        'message': 'Bulk WER evaluation is queued.',
        'result': None,
        'errors': [],
    }

    with BULK_WER_TASKS_LOCK:
        BULK_WER_TASKS[task_id] = task

    return task


def _update_bulk_wer_task(task_id, **updates):
    # Safely update the shared bulk-WER task dictionary from any thread.
    with BULK_WER_TASKS_LOCK:
        task = BULK_WER_TASKS.get(task_id)
        if not task:
            return None

        task.update(updates)
        return dict(task)


def _get_bulk_wer_task(task_id):
    # Return a copy of a task so callers cannot mutate shared state accidentally.
    with BULK_WER_TASKS_LOCK:
        task = BULK_WER_TASKS.get(task_id)
        return dict(task) if task else None


def _run_bulk_wer_task(task_id, audio_batch_folder, transcript_batch_folder, report_csv_path):
    # Run bulk WER in a background thread and publish progress for polling.
    try:
        _update_bulk_wer_task(
            task_id,
            status='running',
            message='Bulk WER evaluation started.',
        )

        def progress_callback(progress):
            updates = {
                'status': 'completed' if progress.get('stage') == 'completed' else 'running',
                'percent': progress.get('percent', 0.0),
                'processed_files': progress.get('processed_files', 0),
                'total_files': progress.get('total_files', 0),
                'current_file': progress.get('current_file'),
                'message': progress.get('message', ''),
            }

            if progress.get('result') is not None:
                updates['result'] = progress.get('result')
                updates['errors'] = progress['result'].get('errors', [])

            _update_bulk_wer_task(task_id, **updates)

        result = evaluate_wer_folder(
            audio_batch_folder,
            transcript_batch_folder,
            progress_callback=progress_callback,
            report_csv_path=report_csv_path,
        )
        _update_bulk_wer_task(
            task_id,
            status='completed',
            percent=100.0,
            processed_files=result.get('total_files', 0),
            total_files=result.get('total_files', 0),
            current_file=None,
            message='Bulk WER evaluation completed.',
            result=result,
            errors=result.get('errors', []),
        )
    except Exception as exc:
        logger.error("Background bulk WER evaluation failed: %s\n%s", exc, traceback.format_exc())
        _update_bulk_wer_task(
            task_id,
            status='failed',
            message=str(exc),
            errors=[{'file_name': 'bulk_evaluation', 'error': str(exc)}],
        )


def _build_accuracy_context(**updates):
    # Merge route-specific values into the accuracy page's default context.
    context = _prepare_accuracy_context()
    context.update(updates)
    return context


def _allowed_evaluation_audio(filename):
    # Validate audio formats accepted by WER evaluation routes.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in WER_ALLOWED_EXTENSIONS


def _save_uploaded_file(file_storage, destination_folder):
    # Save a Werkzeug upload using only its basename and a sanitized filename.
    raw_name = Path(file_storage.filename or '').name
    filename = secure_filename(raw_name)
    if not filename:
        raise ValueError('Invalid uploaded filename.')

    destination_path = os.path.join(destination_folder, filename)
    file_storage.save(destination_path)
    return destination_path


def _safe_send_path(file_path):
    # Send evaluation reports only when the requested path is inside the reports folder.
    resolved_path = Path(file_path).resolve()
    evaluation_root = Path(app.config['EVALUATION_FOLDER']).resolve()

    if evaluation_root not in resolved_path.parents and resolved_path != evaluation_root:
        raise ValueError('Requested file is outside evaluation results.')

    return send_file(resolved_path, as_attachment=True)


def _build_export_transcript_data(entries):
    # Convert browser transcript rows into the structure expected by the exporter.
    transcript_entries = []
    speaker_turn_count = {}
    max_end_time = 0

    for entry in entries:
        timestamp = entry.get('timestamp', '00:00:00')
        timestamp_end = entry.get('timestamp_end', timestamp)
        speaker = entry.get('speaker', 'Speaker 1')
        text = entry.get('text', '')

        transcript_entries.append({
            'timestamp_start': timestamp,
            'timestamp_end': timestamp_end,
            'speaker': speaker,
            'text': text,
            'confidence': 1.0
        })
        speaker_turn_count[speaker] = speaker_turn_count.get(speaker, 0) + 1
        max_end_time = max(max_end_time, _timestamp_to_seconds(timestamp_end))

    return {
        'transcript_entries': transcript_entries,
        'speaker_turn_count': speaker_turn_count,
        'num_speakers': len(speaker_turn_count),
        'total_duration': max_end_time
    }


def _timestamp_to_seconds(timestamp):
    # Best-effort conversion from HH:MM:SS or MM:SS timestamps to seconds.
    try:
        parts = [int(part) for part in str(timestamp).split(':')]
    except ValueError:
        return 0

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return 0


@app.route('/')
def index():
    # Render the main transcription interface.
    return render_template('index.html')


@app.route('/accuracy')
def accuracy_page():
    # Render the model accuracy evaluation page.
    return render_template('accuracy.html', **_prepare_accuracy_context())


@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Receive one audio file, validate it, and store it under a UUID filename.
    logger.info("=== FILE UPLOAD INITIATED ===")

    if 'file' not in request.files:
        logger.warning("No file provided in upload request")
        return jsonify({
            'success': False,
            'error': 'No file provided. Please select an audio file.'
        }), 400

    upload = request.files['file']

    if upload.filename == '':
        logger.warning("Empty filename provided")
        return jsonify({
            'success': False,
            'error': 'No file selected. Please choose an audio file.'
        }), 400

    if not allowed_file(upload.filename):
        logger.warning(f"Invalid file type: {upload.filename}")
        return jsonify({
            'success': False,
            'error': f'Invalid file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        session_id = generate_unique_id()
        original_filename = upload.filename or ''

        file_extension = original_filename.rsplit('.', 1)[1].lower()
        safe_filename = f"{session_id}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

        upload.save(filepath)
        logger.info(f"File saved successfully: {safe_filename}")

        file_info = get_file_info(filepath)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': original_filename,
            'filepath': safe_filename,
            'file_info': file_info,
            'message': 'File uploaded successfully! Ready for processing.'
        }), 200

    except Exception as exc:
        logger.error(f"Upload error: {str(exc)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(exc)}'
        }), 500


@app.route('/api/process', methods=['POST'])
def process_audio():
    # Run Whisper, then Pyannote, then merge the two timelines for the UI.
    logger.info("=== AUDIO PROCESSING INITIATED ===")

    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        filename = data.get('filename')
        model = data.get('model', MODEL_NAME)
        device = data.get('device', 'cpu')

        if model not in SUPPORTED_MODELS:
            logger.warning(f"Unsupported Whisper model '{model}' requested. Falling back to {MODEL_NAME}.")
            model = MODEL_NAME

        if device not in {'cpu', 'cuda'}:
            device = 'cpu'

        if not session_id or not filename:
            return jsonify({
                'success': False,
                'error': 'Missing session_id or filename'
            }), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return jsonify({
                'success': False,
                'error': 'Uploaded file not found'
            }), 404

        logger.info(f"Processing: {filename}")
        logger.info(f"Options: model={model}, device={device}")

        start_time = time.time()

        try:
            audio_path = filepath
            transcription_result = transcribe_audio_file(
                audio_path,
                model_name=model,
                device=device
            )

            if not transcription_result.get('success'):
                logger.error(f"Transcription failed: {transcription_result.get('error')}")
                return jsonify({
                    'success': False,
                    'error': transcription_result.get('error', 'Transcription failed')
                }), 500

            diarization_result = diarize_audio_file(
                audio_path,
                device=device
            )

            if not diarization_result.get('success'):
                logger.warning(f"Diarization failed, falling back to single speaker: {diarization_result.get('error')}")
                # Keep the transcript usable even when Pyannote cannot initialize or run.
                transcript_entries = []
                for seg in transcription_result.get('segments', []):
                    start_sec = seg.get('start', 0.0)
                    mins = int(start_sec // 60)
                    secs = int(start_sec % 60)
                    timestamp = f"{mins:02d}:{secs:02d}"
                    text = seg.get('text', '').strip()
                    transcript_entries.append({
                        'timestamp': timestamp,
                        'speaker': 'Speaker 1',
                        'text': text
                    })
            else:
                merged_result = merge_results(transcription_result, diarization_result)

                if not merged_result.get('success'):
                    logger.error(f"Merge failed: {merged_result.get('error')}")
                    return jsonify({
                        'success': False,
                        'error': merged_result.get('error', 'Failed to merge transcript and diarization')
                    }), 500

                # The browser expects compact rows with timestamp, speaker, and text.
                transcript_entries = []
                for entry in merged_result.get('transcript_entries', []):
                    timestamp = entry.get('timestamp_start', '00:00:00')
                    text = entry.get('text', '').strip()
                    transcript_entries.append({
                        'timestamp': timestamp,
                        'speaker': entry.get('speaker', 'Speaker 1'),
                        'text': text
                    })

            processing_time = time.time() - start_time

            return jsonify({
                'success': True,
                'session_id': session_id,
                'transcript': transcript_entries,
                'processing_time': processing_time,
                'model': transcription_result.get('model', model),
                'device': transcription_result.get('device', device),
                'requested_device': transcription_result.get('requested_device', device)
            }), 200

        except Exception as exc:
            logger.error(f"Processing exception: {str(exc)}\n{traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(exc)}'
            }), 500

    except Exception as exc:
        logger.error(f"Processing error: {str(exc)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(exc)}'
            }), 500


@app.route('/evaluate-wer', methods=['POST'])
def evaluate_wer_route():
    # Evaluate one audio file against one manually written transcript.
    page_errors = []

    try:
        audio_file = request.files.get('audio_file')
        reference_file = request.files.get('reference_file')

        if not audio_file or not audio_file.filename:
            page_errors.append('Please upload one audio file for single-file WER evaluation.')
        elif not _allowed_evaluation_audio(audio_file.filename):
            page_errors.append('Unsupported audio format for WER evaluation. Use WAV, MP3, M4A, or FLAC.')

        if not reference_file or not reference_file.filename:
            page_errors.append('Please upload one reference transcript text file.')
        elif not reference_file.filename.lower().endswith('.txt'):
            page_errors.append('Reference transcript must be a .txt file.')

        if page_errors:
            return render_template('accuracy.html', **_build_accuracy_context(page_errors=page_errors))

        audio_path = _save_uploaded_file(audio_file, WER_AUDIO_FOLDER)
        reference_path = _save_uploaded_file(reference_file, WER_REFERENCE_FOLDER)
        result = evaluate_wer(audio_path, reference_path)

        return render_template('accuracy.html', **_build_accuracy_context(single_wer_result=result))
    except Exception as exc:
        logger.error("Single WER evaluation failed: %s\n%s", exc, traceback.format_exc())
        page_errors.append(str(exc))
        return render_template('accuracy.html', **_build_accuracy_context(page_errors=page_errors))


@app.route('/evaluate-wer-folder', methods=['POST'])
def evaluate_wer_folder_route():
    # Run bulk WER synchronously for browsers without JavaScript polling.
    page_errors = []

    try:
        audio_files = [file for file in request.files.getlist('audio_files') if file and file.filename]
        transcript_files = [file for file in request.files.getlist('transcript_files') if file and file.filename]

        if not audio_files:
            page_errors.append('Please upload audio files or an audio folder for bulk WER evaluation.')
        if not transcript_files:
            page_errors.append('Please upload transcript text files or a transcript folder for bulk WER evaluation.')

        invalid_audio_files = [file.filename for file in audio_files if not _allowed_evaluation_audio(file.filename)]
        invalid_transcripts = [file.filename for file in transcript_files if not file.filename.lower().endswith('.txt')]

        if invalid_audio_files:
            page_errors.append(f"Unsupported bulk audio files: {', '.join(invalid_audio_files)}")
        if invalid_transcripts:
            page_errors.append(f"Unsupported transcript files: {', '.join(invalid_transcripts)}")

        if page_errors:
            return render_template('accuracy.html', **_build_accuracy_context(page_errors=page_errors))

        batch_id = generate_unique_id()
        audio_batch_folder = os.path.join(WER_AUDIO_FOLDER, batch_id)
        transcript_batch_folder = os.path.join(WER_REFERENCE_FOLDER, batch_id)
        os.makedirs(audio_batch_folder, exist_ok=True)
        os.makedirs(transcript_batch_folder, exist_ok=True)

        for file_storage in audio_files:
            _save_uploaded_file(file_storage, audio_batch_folder)

        for file_storage in transcript_files:
            _save_uploaded_file(file_storage, transcript_batch_folder)

        result = evaluate_wer_folder(audio_batch_folder, transcript_batch_folder)
        page_errors.extend(error['error'] for error in result.get('errors', []))

        return render_template(
            'accuracy.html',
            **_build_accuracy_context(bulk_wer_result=result, page_errors=page_errors)
        )
    except Exception as exc:
        logger.error("Bulk WER evaluation failed: %s\n%s", exc, traceback.format_exc())
        page_errors.append(str(exc))
        return render_template('accuracy.html', **_build_accuracy_context(page_errors=page_errors))


@app.route('/evaluate-wer-folder/start', methods=['POST'])
def evaluate_wer_folder_start_route():
    # Start an asynchronous bulk WER run and return its polling task id.
    try:
        audio_files = [file for file in request.files.getlist('audio_files') if file and file.filename]
        transcript_files = [file for file in request.files.getlist('transcript_files') if file and file.filename]

        page_errors = []
        if not audio_files:
            page_errors.append('Please upload audio files or an audio folder for bulk WER evaluation.')
        if not transcript_files:
            page_errors.append('Please upload transcript text files or a transcript folder for bulk WER evaluation.')

        invalid_audio_files = [file.filename for file in audio_files if not _allowed_evaluation_audio(file.filename)]
        invalid_transcripts = [file.filename for file in transcript_files if not file.filename.lower().endswith('.txt')]

        if invalid_audio_files:
            page_errors.append(f"Unsupported bulk audio files: {', '.join(invalid_audio_files)}")
        if invalid_transcripts:
            page_errors.append(f"Unsupported transcript files: {', '.join(invalid_transcripts)}")

        if page_errors:
            return jsonify({'success': False, 'errors': page_errors}), 400

        batch_id = generate_unique_id()
        audio_batch_folder = os.path.join(WER_AUDIO_FOLDER, batch_id)
        transcript_batch_folder = os.path.join(WER_REFERENCE_FOLDER, batch_id)
        os.makedirs(audio_batch_folder, exist_ok=True)
        os.makedirs(transcript_batch_folder, exist_ok=True)

        for file_storage in audio_files:
            _save_uploaded_file(file_storage, audio_batch_folder)

        for file_storage in transcript_files:
            _save_uploaded_file(file_storage, transcript_batch_folder)

        task = _create_bulk_wer_task(total_files=len(audio_files))
        report_csv_path = os.path.join(app.config['EVALUATION_FOLDER'], f'wer_report_{task["task_id"]}.csv')

        worker = threading.Thread(
            target=_run_bulk_wer_task,
            args=(task['task_id'], audio_batch_folder, transcript_batch_folder, report_csv_path),
            daemon=True,
        )
        worker.start()

        return jsonify({
            'success': True,
            'task_id': task['task_id'],
            'status': task['status'],
            'percent': task['percent'],
            'processed_files': task['processed_files'],
            'total_files': task['total_files'],
            'message': task['message'],
        }), 202
    except Exception as exc:
        logger.error("Bulk WER task start failed: %s\n%s", exc, traceback.format_exc())
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/evaluate-wer-folder/status/<task_id>', methods=['GET'])
def evaluate_wer_folder_status_route(task_id):
    # Return progress and final results for an asynchronous bulk WER task.
    task = _get_bulk_wer_task(task_id)
    if task is None:
        return jsonify({'success': False, 'error': 'Bulk WER task not found.'}), 404

    return jsonify({'success': True, 'task': task}), 200


@app.route('/evaluate-der', methods=['POST'])
def evaluate_der_route():
    # Evaluate diarization quality on a small CALLHOME sample set.
    page_errors = []

    try:
        num_samples = request.form.get('num_samples', '3').strip() or '3'
        result = evaluate_der(num_samples=int(num_samples))
        page_errors.extend(error['error'] for error in result.get('errors', []))
        return render_template('accuracy.html', **_build_accuracy_context(der_result=result, page_errors=page_errors))
    except Exception as exc:
        logger.error("DER evaluation failed: %s\n%s", exc, traceback.format_exc())
        page_errors.append(str(exc))
        return render_template('accuracy.html', **_build_accuracy_context(page_errors=page_errors))


@app.route('/evaluation-download')
def evaluation_download():
    # Download generated evaluation reports after path safety checks.
    file_path = request.args.get('path', '')
    if not file_path:
        return jsonify({'success': False, 'error': 'Missing file path.'}), 400

    try:
        return _safe_send_path(file_path)
    except Exception as exc:
        logger.error("Evaluation download failed: %s", exc)
        return jsonify({'success': False, 'error': str(exc)}), 400

@app.route('/api/download/txt/<session_id>', methods=['GET'])
def download_txt(session_id):
    # Legacy placeholder endpoint kept for compatibility with older clients.
    logger.info(f"TXT download requested for session: {session_id}")

    try:
        # The current UI generates TXT downloads in the browser; keep this legacy route stable.
        return jsonify({
            'success': False,
            'message': 'TXT export implemented in Phase 7'
        }), 501  # 501 = Not Implemented

    except Exception as exc:
        logger.error(f"Download error: {str(exc)}")
        return jsonify({
            'success': False,
            'error': f'Download failed: {str(exc)}'
        }), 500

@app.route('/api/download/pdf', methods=['POST'])
def download_pdf_from_transcript():
    # Build a PDF export from transcript rows sent by the browser.
    logger.info("PDF download requested from browser transcript")

    try:
        data = request.get_json(silent=True) or {}
        transcript = data.get('transcript')

        if not isinstance(transcript, list) or not transcript:
            return jsonify({
                'success': False,
                'error': 'No transcript data provided.'
            }), 400

        transcript_data = _build_export_transcript_data(transcript)
        metadata = {
            'title': data.get('title') or 'Meeting Transcript',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'session_id': data.get('session_id', ''),
            'model': data.get('model', ''),
            'device': data.get('device', '')
        }
        pdf_bytes = export_transcript(transcript_data, format='pdf', metadata=metadata)

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='meeting_transcript.pdf'
        )

    except Exception as exc:
        logger.error(f"PDF download error: {str(exc)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'PDF download failed: {str(exc)}'
        }), 500


@app.route('/api/download/pdf/<session_id>', methods=['GET'])
def download_pdf(session_id):
    # Legacy placeholder endpoint kept for compatibility with older clients.
    logger.info(f"PDF download requested for session: {session_id}")

    try:
        # The active PDF export route accepts transcript rows via POST at /api/download/pdf.
        return jsonify({
            'success': False,
            'message': 'PDF export implemented in Phase 7'
        }), 501  # 501 = Not Implemented

    except Exception as exc:
        logger.error(f"Download error: {str(exc)}")
        return jsonify({
            'success': False,
            'error': f'Download failed: {str(exc)}'
        }), 500

@app.errorhandler(400)
def bad_request(error):
    # Handle 400 Bad Request errors.
    logger.warning(f"Bad request: {error}")
    return jsonify({
        'success': False,
        'error': 'Bad request. Please check your input.',
        'details': str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    # Handle 404 Not Found errors.
    logger.warning(f"Resource not found: {error}")
    return jsonify({
        'success': False,
        'error': 'Resource not found.'
    }), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    # Handle 413 Payload Too Large errors (file too big).
    logger.warning(f"File too large: {error}")
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 500MB.'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    # Handle 500 Internal Server Error.
    logger.error(f"Internal server error: {error}\n{traceback.format_exc()}")
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

if __name__ == '__main__':
    # Environment variables make local ports easy to change without editing code.
    app_host = os.getenv('HOST', '0.0.0.0')
    app_port = int(os.getenv('PORT', '8000'))

    print("\n" + "="*60)
    print("Whisper-Based Multi-Speaker Meeting Transcription System - Starting Server")
    print("="*60)
    print("\n✓ Flask application initialized")
    print("✓ Upload folder:", app.config['UPLOAD_FOLDER'])
    print("✓ Output folder:", app.config['OUTPUT_FOLDER'])
    print("✓ Max file size: 500MB")
    print(f"\n📍 Access the application at: http://localhost:{app_port}")
    print("\n🔴 Press CTRL+C to stop the server")
    print("="*60 + "\n")
    app.run(
        debug=False,
        threaded=True,
        host=app_host,
        port=app_port
    )
