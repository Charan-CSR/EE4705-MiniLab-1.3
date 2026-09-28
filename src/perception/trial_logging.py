"""One auditable row and a few camera snapshots per Task 4 attempt."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from PIL import Image


FIELDS = (
    'trial_id', 'run_label', 'started_utc', 'mode', 'typed_command', 'provider',
    'parse_latency_s', 'target_class', 'target_color', 'start_x_m', 'start_y_m',
    'start_yaw_rad', 'end_x_m', 'end_y_m', 'elapsed_s', 'status', 'reason',
    'search_occurred', 'unique_camera_frames', 'target_detected_frames',
    'target_detected_any', 'first_frame_target_detected', 'first_target_confidence',
    'final_frame_target_detected', 'final_distance_m', 'within_0_80_m',
    'initial_image', 'first_target_image', 'final_image',
    'detection_correct_manual', 'grounding_correct_manual',
    'contact_free_manual', 'notes_manual',
)


class TrialRecorder:
    def __init__(self, project_root, run_label='development'):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', run_label):
            raise ValueError('trial run must use 1-50 letters, digits, underscores or hyphens')
        self.run_label = run_label
        self.directory = Path(project_root) / 'results' / 'task4' / 'trials' / run_label
        self.current = None

    def begin(self, target, start_pose, now, mode, context=None):
        context = context or {}
        trial_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:6]
        self.current = {
            'id': trial_id, 'target': target, 'start_pose': start_pose,
            'started': now, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'mode': mode, 'context': context, 'last_stamp': None,
            'frames': 0, 'hits': 0, 'first_hit_conf': '',
            'first_hit': None, 'first': None, 'last': None,
            'first_visible': '', 'final_visible': '', 'search': False,
        }

    def observe(self, observation, now):
        record = self.current
        if (record is None or observation is None
                or observation.captured < record['started']
                or now - observation.captured > 0.8
                or observation.captured == record['last_stamp']):
            return
        record['last_stamp'] = observation.captured
        record['frames'] += 1
        matches = [d for d in observation.detections
                   if (d.class_name, d.color) == record['target']]
        visible = bool(matches)
        record['final_visible'] = int(visible)
        if visible:
            record['hits'] += 1
        frame = getattr(observation, 'rgb_frame', None)
        if record['first_visible'] == '':
            record['first_visible'] = int(visible)
            if frame is not None:
                record['first'] = frame.copy()
        if visible and record['first_hit_conf'] == '':
            record['first_hit_conf'] = round(max(d.confidence for d in matches), 3)
            if frame is not None:
                record['first_hit'] = frame.copy()
        if frame is not None:
            record['last'] = frame.copy()

    def mark_search(self):
        if self.current is not None:
            self.current['search'] = True

    def finish(self, status, reason, now, end_pose, final_distance=None):
        record = self.current
        if record is None:
            return
        self.current = None
        self.directory.mkdir(parents=True, exist_ok=True)
        pictures = {}
        for name, frame in (('initial', record['first']),
                            ('first_target', record['first_hit']),
                            ('final', record['last'])):
            if frame is not None:
                filename = f"{record['id']}_{name}.png"
                Image.fromarray(frame).save(self.directory / filename)
                pictures[name] = filename
        context = record['context']
        row = dict.fromkeys(FIELDS, '')
        row.update({
            'trial_id': record['id'], 'run_label': self.run_label,
            'started_utc': record['started_utc'], 'mode': record['mode'],
            'typed_command': context.get('command', ''),
            'provider': context.get('provider', ''),
            'parse_latency_s': context.get('parse_latency_s', ''),
            'target_class': record['target'][0], 'target_color': record['target'][1],
            'start_x_m': round(record['start_pose'][0], 3),
            'start_y_m': round(record['start_pose'][1], 3),
            'start_yaw_rad': round(record['start_pose'][2], 3),
            'end_x_m': round(end_pose[0], 3), 'end_y_m': round(end_pose[1], 3),
            'elapsed_s': round(max(0, now - record['started']), 2),
            'status': status, 'reason': reason,
            'search_occurred': int(record['search']),
            'unique_camera_frames': record['frames'],
            'target_detected_frames': record['hits'],
            'target_detected_any': int(record['hits'] > 0),
            'first_frame_target_detected': record['first_visible'],
            'first_target_confidence': record['first_hit_conf'],
            'final_frame_target_detected': record['final_visible'],
            'final_distance_m': '' if final_distance is None else round(final_distance, 3),
            'within_0_80_m': '' if final_distance is None else int(final_distance <= 0.80),
            'initial_image': pictures.get('initial', ''),
            'first_target_image': pictures.get('first_target', ''),
            'final_image': pictures.get('final', ''),
        })
        csv_path = self.directory / 'missions.csv'
        is_new = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        print(f'[TRIAL] saved={csv_path} id={record["id"]}', flush=True)
