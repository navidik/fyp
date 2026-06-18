# Combine Whisper text segments with Pyannote speaker segments.
import logging
from datetime import datetime
from typing import Dict, List


logger = logging.getLogger(__name__)

MERGE_CONFIG = {
    'min_overlap_threshold': 0.1,
    'group_threshold': 0.5,
    'include_silence': False,
    'confidence_threshold': 0.3,
}


class TranscriptMerger:
    # Assign speakers to transcript segments and group consecutive turns.
    def __init__(self, config=None):
        # Merge caller-provided tuning values with the default merge configuration.
        self.config = {**MERGE_CONFIG, **(config or {})}
        logger.info("TranscriptMerger initialized")
        logger.info(f"Config: {self.config}")

    def merge(self, transcription_result: Dict, diarization_result: Dict) -> Dict:
        # Validate model outputs, align them by time, and return export-ready rows.
        logger.info("Starting merge operation")

        try:
            if not transcription_result or not transcription_result.get('success'):
                logger.error("Invalid transcription result")
                return {
                    'success': False,
                    'error': 'Invalid transcription result'
                }

            if not diarization_result or not diarization_result.get('success'):
                logger.error("Invalid diarization result")
                return {
                    'success': False,
                    'error': 'Invalid diarization result'
                }

            text_segments = transcription_result.get('segments', [])
            speaker_segments = diarization_result.get('segments', [])
            logger.info(f"Merging {len(text_segments)} text segments with {len(speaker_segments)} speaker segments")

            assigned_segments = self._assign_speakers(text_segments, speaker_segments)
            grouped_entries = self._group_by_speaker(assigned_segments)
            speaker_counts = self._count_speaker_turns(grouped_entries)

            full_transcript = '\n\n'.join([
                f"[{entry['timestamp_start']} - {entry['timestamp_end']}]\n"
                f"{entry['speaker']}:\n{entry['text']}"
                for entry in grouped_entries
            ])

            total_duration = max(
                [seg.get('end_time', 0) for seg in speaker_segments]
                + [seg.get('end', 0) for seg in text_segments],
                default=0
            )
            result = {
                'success': True,
                'full_transcript': full_transcript,
                'transcript_entries': grouped_entries,
                'speaker_turn_count': speaker_counts,
                'total_duration': total_duration,
                'num_speakers': len(speaker_counts),
                'merged_timestamp': self._get_timestamp()
            }
            logger.info(f"✓ Merge successful: {len(grouped_entries)} entries, {len(speaker_counts)} speakers")
            return result

        except Exception as exc:
            logger.error(f"Merge error: {str(exc)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Merge failed: {str(exc)}'
            }

    def _assign_speakers(self, text_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        # Pick the diarization segment with the largest overlap for each text segment.
        logger.info("Assigning speakers to text segments")
        assigned = []
        for text_segment in text_segments:
            text_start = text_segment.get('start', 0)
            text_end = text_segment.get('end', 0)
            text_duration = text_end - text_start
            best_speaker = None
            best_overlap = 0
            best_confidence = 0

            for speaker_seg in speaker_segments:
                speaker_start = speaker_seg.get('start_time', 0)
                speaker_end = speaker_seg.get('end_time', 0)
                overlap_start = max(text_start, speaker_start)
                overlap_end = min(text_end, speaker_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > 0:
                    confidence = overlap_duration / text_duration if text_duration > 0 else 0
                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_speaker = speaker_seg
                        best_confidence = confidence

            assigned_seg = text_segment.copy()
            if best_speaker and best_confidence >= self.config['confidence_threshold']:
                assigned_seg['speaker'] = best_speaker['speaker']
                assigned_seg['speaker_id'] = best_speaker['speaker_id']
                assigned_seg['confidence'] = best_confidence
            else:
                assigned_seg['speaker'] = 'Unknown'
                assigned_seg['speaker_id'] = -1
                assigned_seg['confidence'] = 0.0
            assigned.append(assigned_seg)
        logger.info(f"Assigned speakers to {len(assigned)} segments")
        return assigned

    def _group_by_speaker(self, assigned_segments: List[Dict]) -> List[Dict]:
        # Join adjacent text segments from the same speaker into readable turns.
        logger.info(f"Grouping {len(assigned_segments)} segments by speaker")
        if not assigned_segments:
            return []

        grouped = []
        current_entry = None
        for seg in assigned_segments:
            speaker = seg.get('speaker', 'Unknown')
            speaker_id = seg.get('speaker_id', -1)
            text = seg.get('text', '').strip()

            if not text:
                continue

            if current_entry and current_entry['speaker'] == speaker:
                current_entry['text'] += ' ' + text
                current_entry['end_time'] = seg.get('end', current_entry['end_time'])
                current_entry['segment_ids'].append(seg.get('id', len(grouped)))
                conf = seg.get('confidence', 0)
                current_entry['confidence'] = (
                    (current_entry['confidence'] * (len(current_entry['segment_ids']) - 1) + conf) /
                    len(current_entry['segment_ids'])
                )
            else:
                if current_entry:
                    grouped.append(current_entry)
                current_entry = {
                    'speaker': speaker,
                    'speaker_id': speaker_id,
                    'text': text,
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'confidence': seg.get('confidence', 0),
                    'segment_ids': [seg.get('id', 0)]
                }

        if current_entry:
            grouped.append(current_entry)

        final_grouped = []
        for entry in grouped:
            entry['timestamp_start'] = self._seconds_to_timestamp(entry['start_time'])
            entry['timestamp_end'] = self._seconds_to_timestamp(entry['end_time'])
            entry['duration'] = entry['end_time'] - entry['start_time']
            final_grouped.append(entry)
        logger.info(f"Grouped into {len(final_grouped)} speaker turns")
        return final_grouped

    def _count_speaker_turns(self, grouped_entries: List[Dict]) -> Dict[str, int]:
        # Count how many grouped turns each speaker has in the final transcript.
        counts = {}
        for entry in grouped_entries:
            speaker = entry.get('speaker', 'Unknown')
            counts[speaker] = counts.get(speaker, 0) + 1
        return counts

    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        # Convert seconds into HH:MM:SS while guarding against negative values.
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _get_timestamp() -> str:
        # Timestamp when the merged transcript was created.
        return datetime.now().isoformat()


def merge_results(transcription_result: Dict, diarization_result: Dict) -> Dict:
    # Convenience function used by Flask to merge one transcription run.
    try:
        merger = TranscriptMerger()
        return merger.merge(transcription_result, diarization_result)
    except Exception as exc:
        logger.error(f"Error: {str(exc)}")
        return {
            'success': False,
            'error': str(exc)
        }
