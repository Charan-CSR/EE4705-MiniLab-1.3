# Task 4 handover — 28 September 2026

## Current status

Marie is handing over the implementation and development evidence for further tuning. Gemini natural-language commands can trigger detection, alignment, forward approach and stopping for the red cup, white cup and green bowl. Some initially off-screen targets also succeed. Search from headings facing empty walls remains unreliable. Browser loading/reset responsiveness has also been intermittent.

This is a development handover, not a claim that all assignment requirements or final evaluation are complete. Task 1 is outside this handover.

## Running the current implementation

Mac M1, arm64, Conda environment `ee4705`, VS Code. From the repository root:

```bash
conda activate ee4705
python quadruped_mujoco/eg/play.py --gui --gui-port 8766 --duration 3600 --task4 --trial-run teammate_diagnostic_01
```

Open `http://127.0.0.1:8766/`, keep the camera panel open, select Gemini, and use `yolo_lab`. Example commands: `Go to the red cup`, `Go to the white cup`, `Go to the green bowl`. Reset before independent trials and document starting view/heading. There is no green cup in the scene. Each teammate supplies their own API key locally; do not commit `.env`.

The browser must be receiving the stream for the supplied runtime to render fresh camera frames. The simulator also exits after the configured 3600-second session duration. `--task4-test` bypasses the LLM for development. `--task4-align-only` checks alignment without approach.

## Files and settings to preserve

- `src/perception/detector.py`: YOLOv8s CPU, input `imgsz=960`, confidence 0.25, cup/bowl classes and HSV color grounding. Input is uint8 RGB; converted to BGR for YOLO.
- `src/perception/goto_object.py`: camera worker; SEARCH, ALIGN, APPROACH and VERIFY logic; alternating search patch; stop evaluator and trial logging integration.
- `src/perception/trial_logging.py`: one CSV row per attempt, initial/first-target/final RGB snapshots, manual review fields.
- `quadruped_mujoco/eg/play.py`: simulator integration, LLM command queue, CLI flags, logging context and browser control integration.
- `config/task4_objects.json`: scene-object XY positions for final distance evaluation. Steering must remain independent of these positions.
- Scene changes used during development: camera FOV 55 in `quadruped_mujoco/src/runtime_control/integration.py`, cup mesh scale changed from .01 to .02 in `quadruped_mujoco/src/runtime_control/maps/yolo_lab.xml`. Confirm these changes in the working tree and disclose them in evaluation.
- Visual stopping height fractions: cup .32, bowl .14. Forward commands .40 far and .30 near. Alignment yaw magnitude .30–.45; search .35. Overall mission timeout 60 seconds.
- Latest suggested UI diagnostic: final loop sleep `time.sleep(max(0.001, simulation_dt - elapsed))`. Confirm it was installed and whether it helped; it may slow simulation relative to wall time. It is not a demonstrated fix for a blocked lock.
- Runtime assets include `quadruped_mujoco/eg/model_3400.onnx` and YOLOv8s weights. Preserve existing repository conventions for model files rather than adding duplicate weights.

## Evidence in the supplied missions.csv

The uploaded `evaluation_02/missions.csv` has **9 rows**, although 10 trials were reported verbally. Locate any missing attempt before claiming a ten-trial set. Do not invent a row or remove failures.

| Target | Attempts | Successes | Other outcomes |
|---|---:|---:|---|
| Red cup | 2 | 2 | — |
| White cup | 4 | 2 | 2 search_sweep_exhausted |
| Green bowl | 2 | 2 | — |
| Green cup | 1 | 0 | unsupported_scene_target |
| Total | 9 | 6 | 3 failures |

Across these recorded rows, success is 6/9 (66.7%). If reporting the supported-target subset separately, it is 6/8 (75%); explicitly disclose that the rejected green-cup request is excluded from that denominator. These are descriptive counts from mixed development runs, not a fixed-version benchmark.

Successful navigation times: 25.53, 9.16, 19.47, 27.15, 23.95 and 41.66 seconds; mean 24.49 seconds. Successful final distances range .569–.670 m. These navigation times exclude Gemini parse latency, which has its own column. Three successes have `search_occurred=1` and no target detection in their first sampled frame; inspect their initial images to establish actual initial visibility.

CSV timestamps span runs before and after the bidirectional search patch. The logger does not record commit IDs or configuration hashes. Reconstruct version boundaries from notes/history and keep subsequent evaluation in a distinct run folder.

Manual detection, grounding and contact-review fields are blank. Frame hit counts are model observations, not independently verified accuracy. Review saved PNGs and available video, annotate what can be established, and leave unknowns explicit. Still images alone cannot prove a contact-free trajectory.

## Confirmed search limitation and tuning priorities

The failed white-cup mission `20260928T035010659051Z_84fc21` remained in SEARCH without a matching white-cup detection. Each of six legs ended with `reason=leg_time_limit`. Recorded offsets reached approximately +21.0 degrees and -48.4 degrees: only about 69.4 degrees of heading span, not the intended broad sweep. Seeing the red cup did not trigger target acquisition in this log. The controller filters for white cup; the reversal was commanded by the sweep time cap.

1. **Make search expand actual heading coverage.** Current goals relative to initial search heading are +35, -35, +90, -90, +180, -180 degrees (mirrored when appropriate). Four-second first-leg limits and ten-second later limits cause reversals before those goals are reached. Prefer heading-based completion; use a separate measured-progress watchdog for a stalled turn. Keep a finite overall mission limit and do not equate elapsed time with angular coverage. A short local search followed by a systematic sweep is an alternative to repeatedly crossing the same small interval.
2. **Measure achievable turning before selecting limits.** Log commanded yaw, actual yaw change/rate, frame age, inference latency and wall versus simulated time. At observed slow turn rates, broad coverage plus approach may not fit 60 seconds. Adjust only within assignment constraints. Do not simply increase speed or timeout without measuring the effect on detection, stability and stopping.
3. **Check perception timing at 960.** The saved weak bowl frame improved from .089 confidence at 640 to .261 at 960, but 960 CPU inference took roughly .53 seconds in one test. Existing .8-second stale-observation protection sometimes zeroes commands. Profile this interaction; preserve a stale-frame stop rather than blindly allowing old detections to drive motion.
4. **Test search with distractors and true negative scenes.** Keep class and color matching. Cover targets just outside either camera edge, opposite-facing starts, target dropout/reacquisition and absent targets. Prefer visual last-seen direction when available; do not hard-code a turn direction based on target color or world position.
5. **Investigate browser/reset separately.** Root page returned HTTP 200 promptly; camera sometimes returned 204 before streaming. Both browsers sometimes stalled. If reset freezes after success, inspect browser request handling, reset callbacks, simulation liveness and lock ordering; collect thread stacks if appropriate. The sleep change is unconfirmed. `--gui` uses the browser, so native-viewer `launch_passive` advice does not directly diagnose this mode.
6. **Freeze the final version for evaluation.** Record commit ID and settings, run the specified number of trials, document initial poses and visibility, retain failures and manually score the required metrics. Use a new output folder after each controller change.

## Git handover contents

Commit source/configuration/scene changes, this handover, `missions.csv`, associated evidence PNGs and relevant terminal logs. Retain diagnostic folders as clearly labeled development evidence where useful. Keep `.env`, API keys, Python caches and local environments out of the commit. Use the group's existing policy for videos and large model assets.

Check contributor details before committing: the uploaded navigator header contains a placeholder student number; correct it using Marie's actual details or omit the number. Keep AI-assistance attribution accurate and follow the course disclosure rules.
