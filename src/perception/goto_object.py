"""Task 4 camera-only search/centering/approach, with separate stop evaluation.

Contributor: Marie Chansuphan (A0287345U). 
AI assistance: drafted with ChatGPT; reviewed and tested by the contributor.
"""
import json
import math
from pathlib import Path
import threading
import time
from dataclasses import dataclass


@dataclass
class Observation:
    captured: float
    width: int
    height: int
    detections: list
    rgb_frame: object = None


class PerceptionWorker:
    """Consume only Task 2's front RGB buffer; never access live MuJoCo state.

    The supplied runtime renders only while its browser has a stream subscriber.
    A stale frame therefore stops navigation when the browser stops requesting.
    """
    def __init__(self, runtime):
        self.runtime = runtime
        self.latest = None
        self.error = None
        self.ready = False
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            # Cap CPU threads so YOLO stays under the 0.8 s stale limit
            import torch
            torch.set_num_threads(4)
            from src.perception.detector import ObjectDetector
            detector = ObjectDetector()
            last = None
            while not self.closed.is_set():
                with self.runtime.render_lock:
                    frame = self.runtime.latest_front_frame
                    stamp = self.runtime.last_front_camera_time
                    # Each render publishes a new array, even when pixels match.
                    if frame is None or frame is last:
                        rgb = None
                    else:
                        last = frame
                        rgb = frame.copy()
                if rgb is None:
                    self.closed.wait(0.02)
                    continue
                detections = detector.detect(rgb)
                observation = Observation(stamp, rgb.shape[1], rgb.shape[0], detections, rgb)
                with self.lock:
                    self.latest = observation
                    self.ready = True
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def read(self):
        with self.lock:
            return self.latest

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)


class VisualNavigator:
    """Nonblocking controller. step() returns normalized (vx, vy, wz).

    Commands use image-center error only. Apparent bbox height is the initial
    visual stopping cue. Ground-truth proximity is evaluated AFTER this cue,
    never used to correct a velocity or to continue an approach.
    """
    def __init__(self, align_only=False, timeout=60.0):
        self.align_only = align_only
        self.timeout = timeout
        self.active = False
        self.outcome = None
        self.target = None

    def start(self, class_name, color, now, yaw):
        self.search_direction = 1.0
        self.target = (class_name, color)
        self.active = True
        self.outcome = None
        self.started = now
        self.previous_yaw = yaw
        self.heading = 0.0  # Unwrapped yaw relative to mission start.
        self.previous_step = now
        self.search_goals = None
        self.search_leg = 0
        self.search_sign = 1.0
        self.search_turn_speed = 0.45
        self.search_progress_time = 0.0
        self.search_progress_heading = 0.0
        self.search_end_since = None
        self.search_context_samples = []
        self.search_probe_extended = False
        self.recovery_pending = False
        self.recovery = None
        self.state = 'WAIT'
        self.last_capture = -1.0
        self.last_seen = now
        self.stable = 0
        self.command = (0.0, 0.0, 0.0)
        self.last_log = -math.inf
        self.stop_since = None
        self.stop_capture = None
        self.verification = None
        print(f'[EXEC] goto_object class={class_name} color={color} '
              f'mode={"align_only" if self.align_only else "approach"}', flush=True)

    def finish(self, status, reason):
        self.active = False
        self.outcome = status
        self.command = (0.0, 0.0, 0.0)
        print(f'[MISSION] status={status} reason={reason}', flush=True)
        callback = getattr(self, 'on_finish', None)
        if callback is not None:
            try:
                callback(status, reason)
            except Exception as exc:
                print(f'[TRIAL ERROR] {type(exc).__name__}: {exc}', flush=True)
        return self.command

    def step(self, now, yaw, observation):
        zero = (0.0, 0.0, 0.0)
        if not self.active:
            return zero
        delta = (yaw - self.previous_yaw + math.pi) % (2 * math.pi) - math.pi
        self.previous_yaw = yaw
        self.heading += delta
        dt = max(0.0, now - self.previous_step)
        self.previous_step = now
        if self.state == 'SEARCH' and self.command[2] != 0 and self.recovery is None:
            self.search_progress_time += dt
        if now - self.started >= self.timeout:
            return self.finish('FAIL', 'timeout')
        if (observation is None or observation.captured < self.started
                or now - observation.captured > 0.8):
            self.command = zero
            if now - self.last_seen > 5:
                return self.finish('FAIL', 'camera_or_detection_stale')
            return zero
        if observation.captured == self.last_capture:
            return self.command
        self.last_capture = observation.captured
        self.last_seen = now
        candidates = [d for d in observation.detections
                      if (d.class_name, d.color) == self.target]
        if not candidates:
            # Non-target objects are search context only, never arrival targets.
            # Require repeated fresh observations before extending a local probe.
            context_labels = sorted({f'{d.color}_{d.class_name}' for d in observation.detections
                                     if d.class_name in ('cup', 'bowl')
                                     and d.color != 'unknown' and d.confidence >= 0.25})
            if context_labels:
                self.search_context_samples.append((observation.captured, context_labels))
            self.search_context_samples = [s for s in self.search_context_samples
                                           if now - s[0] <= 2.0]
            self.stable = 0
            self.command = zero
            if self.state == 'VERIFY':
                if now - self.stop_since > 3:
                    return self.finish('FAIL', 'target_not_visible_at_stop')
                return zero
            if self.state != 'LOST' and self.state != 'SEARCH':
                self.missing_since = now
                self.state = 'LOST'
            if self.state == 'LOST' and now - self.missing_since < 0.5:
                return zero
            if self.state != 'SEARCH':
                self.state = 'SEARCH'
                if self.search_goals is None:
                    # Short local probe, then a continuous sweep the other way.
                    # Advance by measured yaw, never by an elapsed leg timer.
                    self.search_sign = self.search_direction
                    self.search_goals = [self.heading + self.search_direction * math.radians(a)
                                         for a in (25, -335)]
                self.search_progress_time = 0.0
                self.search_progress_heading = self.heading
                if self.recovery_pending:
                    self.recovery = (now, self.heading, self.search_direction)
                    self.recovery_pending = False
                print(f'[SEARCH] class={self.target[0]} color={self.target[1]} '
                      f'heading_sweep leg={self.search_leg + 1} '
                      f'wz_limit={self.search_turn_speed:.2f}', flush=True)
            return self._search_command(now)
        target = max(candidates, key=lambda d: d.confidence)
        x1, y1, x2, y2 = target.bbox
        error = ((x1 + x2) / 2 - observation.width / 2) / (observation.width / 2)
        if abs(error) > 0.10:
            self.search_direction = -1.0 if error > 0 else 1.0
        self.recovery_pending = True
        self.recovery = None
        fraction = (y2 - y1) / observation.height
        if now - self.last_log > 0.5:
            print(f'[DETECT] class={target.class_name} color={target.color} '
                  f'conf={target.confidence:.2f} bbox={list(target.bbox)} '
                  f'center_error={error:.3f} height_fraction={fraction:.3f}', flush=True)
            self.last_log = now
        if self.state == 'VERIFY':
            # Zero commands have been held; require a newly captured frame.
            self.command = zero
            if now - self.stop_since >= 0.5 and observation.captured > self.stop_since + 0.3:
                self.verification = target
            return zero
        # Preserve the direction selected from the last observed target.
        alignment_limit = 0.16 if self.stable > 0 else 0.10
        if abs(error) > alignment_limit:
            self.state = 'ALIGN'
            self.stable = 0
            wz = -math.copysign(min(0.45, max(0.30, 0.8 * abs(error))), error)
            self.command = (0.0, 0.0, wz)
            return self.command
        self.stable += 1
        self.command = zero
        if self.stable < 3:
            self.state = 'ALIGN'
            return zero
        if self.align_only:
            # This intentionally does NOT claim a completed Task 4 mission.
            print('[ALIGNED] target centered; approach disabled for this test', flush=True)
            return self.finish('TEST_COMPLETE', 'alignment_only')
        # Initial cue for this scene's doubled cups and original bowl at fovy=55.
        # Must calibrate using evaluation distances; changing size/FOV invalidates it.
        threshold = {'cup': 0.32, 'bowl': 0.14}[self.target[0]]
        if fraction >= threshold:
            self.state = 'VERIFY'
            self.stop_since = now
            print('[STOP] reason=visual_size_threshold; checking fresh detection', flush=True)
            return zero
        self.state = 'APPROACH'
        # Slow forward command; no forward motion while substantially misaligned.
        self.command = (0.40 if fraction < threshold * 0.75 else 0.30, 0.0, 0.0)
        return self.command

    def _search_command(self, now):
        # Briefly look toward the last sighting after a detection dropout.
        # Sweep goals survive reacquisition/dropout cycles.
        if self.recovery is not None:
            since, heading, direction = self.recovery
            if now - since < 2.0 and direction * (self.heading - heading) < math.radians(20):
                self.command = (0.0, 0.0, self.search_turn_speed * direction)
                return self.command
            self.recovery = None
            self.search_progress_time = 0.0
            self.search_progress_heading = self.heading
        while self.search_leg < len(self.search_goals):
            goal = self.search_goals[self.search_leg]
            direction = self.search_sign if self.search_leg == 0 else -self.search_sign
            # Directed crossing tolerates yaw overshoot without reversing back.
            reached = direction * (self.heading - goal) >= 0.0
            if reached:
                recent = [s for s in self.search_context_samples if now - s[0] <= 2.0]
                if (self.search_leg == 0 and not self.search_probe_extended
                        and len(recent) >= 2):
                    # Extend once from 25 to 90 degrees when the camera offers
                    # evidence of objects. This is a heuristic, not knowledge of
                    # the target's direction. Keep a bounded full-sweep fallback.
                    self.search_probe_extended = True
                    extension = self.search_sign * math.radians(65)
                    self.search_goals = [g + extension for g in self.search_goals]
                    self.search_progress_time = 0.0
                    self.search_progress_heading = self.heading
                    labels = ','.join(sorted({label for _, labels in recent for label in labels}))
                    print(f'[SEARCH] reason=visible_object_context objects={labels} '
                          f'probe_extended_to_deg=90', flush=True)
                    continue
                print(f'[SEARCH] leg={self.search_leg + 1} '
                      f'reason=heading_reached '
                      f'yaw_offset_deg={math.degrees(self.heading):.1f}', flush=True)
                self.search_leg += 1
                self.search_progress_time = 0.0
                self.search_progress_heading = self.heading
                continue
            # Separate stalled-turn watchdog: fail rather than pretending a
            # sweep finished. Time counts only while commanding a search turn.
            if self.search_progress_time >= 8.0:
                progress = direction * (self.heading - self.search_progress_heading)
                print(f'[SEARCH] progress_deg={math.degrees(progress):.1f} '
                      f'commanded_seconds={self.search_progress_time:.1f}', flush=True)
                if progress < math.radians(3):
                    return self.finish('FAIL', 'search_turn_not_progressing')
                self.search_progress_time = 0.0
                self.search_progress_heading = self.heading
            self.command = (0.0, 0.0, self.search_turn_speed * direction)
            return self.command
        self.command = (0.0, 0.0, 0.0)
        # Allow an observation captured after reaching the final heading before
        # declaring no target. step() checks matching detections first.
        if self.search_end_since is None:
            self.search_end_since = now
        elif self.last_capture > self.search_end_since:
            return self.finish('FAIL', 'search_sweep_without_target')
        return self.command


class Task4Bridge:
    """Simulation-thread adapter; the evaluator alone reads ground-truth XY."""
    def __init__(self, runtime, model, data, align_only=False,
                 run_label='development', mode='direct_test'):
        from src.perception.trial_logging import TrialRecorder
        self.runtime, self.model, self.data = runtime, model, data
        self.mode = mode
        self.recorder = TrialRecorder(Path(__file__).resolve().parents[2], run_label)
        self.worker = PerceptionWorker(runtime)
        self.navigator = VisualNavigator(align_only=align_only)
        self.navigator.on_finish = self._record_finish
        self.hold = False
        self.failure_pending = False
        self.settle_deadline = None
        # Task 2 yolo_lab object-center XY; evaluation ONLY, never sent to step().
        config_path = Path(__file__).resolve().parents[2] / 'config/task4_objects.json'
        objects = json.loads(config_path.read_text(encoding='utf-8'))['objects']
        self.evaluation_xy = {(o['class_name'], o['color']): tuple(o['xy']) for o in objects}


    @property
    def active(self):
        return self.navigator.active

    @property
    def busy(self):
        return self.active

    def start(self, class_name, color, yaw, context=None):
        self.hold = True
        self.failure_pending = False
        self.settle_deadline = None
        now = time.monotonic()
        base = self.data.body('trunk').xpos
        self.recorder.begin((class_name, color), (float(base[0]), float(base[1]), yaw),
                            now, self.mode, context)
        self.navigator.start(class_name, color, now, yaw)
        if self.runtime.maps.active_map != 'yolo_lab':
            self.navigator.finish('FAIL', 'requires_yolo_lab_scene')
            self.failure_pending = True
        elif (class_name, color) not in self.evaluation_xy:
            self.navigator.finish('FAIL', 'unsupported_scene_target')
            self.failure_pending = True

    def cancel(self, reason='stopped'):
        if self.active:
            self.navigator.finish('FAIL', reason)
        self.hold = True

    def release(self):
        self.hold = False

    def command(self, yaw):
        import mujoco
        now = time.monotonic()
        observation = self.worker.read()
        self.recorder.observe(observation, now)
        if self.worker.error and self.active:
            self.navigator.finish('FAIL', 'perception_error:' + self.worker.error)
            self.failure_pending = True
        was_active = self.active
        command = self.navigator.step(now, yaw, observation)
        if self.navigator.state == 'SEARCH':
            self.recorder.mark_search()
        if self.active and now - getattr(self, "_last_debug", 0.0) >= 1.0:
            self._last_debug = now
            speed = math.hypot(float(self.data.qvel[0]), float(self.data.qvel[1]))
            print(
                f"[NAV] state={self.navigator.state} "
                f"vx_cmd={command[0]:.2f} wz_cmd={command[2]:.2f} "
                f"actual_speed={speed:.3f} m/s "
                f"yaw_offset_deg={math.degrees(self.navigator.heading):.1f}",
                flush=True,
            )
        # Abort on contact between the robot and non-floor lab geometry.
        if self.active:
            for contact in self.data.contact[:self.data.ncon]:
                names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ''
                         for g in (contact.geom1, contact.geom2)]
                is_map = [n.startswith('map_') for n in names]
                if sum(is_map) == 1:
                    name = names[is_map.index(True)]
                    if any(word in name for word in ('table', 'bench', 'wall')):
                        self.navigator.finish('FAIL', 'obstacle_contact')
                        command = (0.0, 0.0, 0.0)
                        break
        if self.navigator.verification is not None and self.active:
            # This branch cannot steer: navigation has already entered VERIFY.
            speed = math.hypot(float(self.data.qvel[0]), float(self.data.qvel[1]))
            if speed > 0.08:
                if self.settle_deadline is None:
                    self.settle_deadline = now + 3
                if now > self.settle_deadline:
                    self.navigator.finish('FAIL', 'robot_did_not_settle')
                self.navigator.verification = None
                return (0.0, 0.0, 0.0)
            base = self.data.body('trunk').xpos
            xy = self.evaluation_xy[self.navigator.target]
            distance = math.hypot(float(base[0]) - xy[0], float(base[1]) - xy[1])
            print(f'[EVAL] final_distance={distance:.3f} m', flush=True)
            if distance <= 0.80:
                cls, color = self.navigator.target
                print(f'[FOUND] class={cls} color={color} '
                      f't={now-self.navigator.started:.1f} s d={distance:.3f} m', flush=True)
                self.navigator.finish('SUCCESS', 'visual_stop_and_proximity_verified')
            else:
                self.navigator.finish('FAIL', 'visual_stop_outside_0.80m')
        if was_active and not self.active and self.navigator.outcome == 'FAIL':
            self.failure_pending = True
        return command if self.active else (0.0, 0.0, 0.0)

    def close(self):
        self.cancel('shutdown')
        self.worker.close()

    def _record_finish(self, status, reason):
        base = self.data.body('trunk').xpos
        target = self.navigator.target
        xy = self.evaluation_xy.get(target)
        distance = (math.hypot(float(base[0]) - xy[0], float(base[1]) - xy[1])
                    if xy is not None else None)
        if self.navigator.state == 'SEARCH':
            self.recorder.mark_search()
        self.recorder.finish(status, reason, time.monotonic(),
                             (float(base[0]), float(base[1])), distance)
