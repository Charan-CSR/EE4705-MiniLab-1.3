# Task 4 integration checkpoint

This is an initial integration built against the play.py uploaded on 25 September.
Keep the existing Task 2 scene modifications (cup mesh scale 0.02, front FOV 55).
Keep the detector.py already tested. No API key is included or required for the
first direct test. Add contributor names/matriculation numbers to source files.

## First test: search and alignment only

    python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4-test --task4-align-only

Open http://127.0.0.1:8765 and KEEP THE BROWSER OPEN. Select yolo_lab. The worker
uses the dedicated front RGB buffer regardless of the displayed camera.
Wait for the scene to appear, then type this into the simulation terminal:

    goto cup red

The robot should turn toward the red cup with no commanded forward velocity.
Some physical drift can occur while turning. Expect [DETECT], [ALIGNED], and
[MISSION] status=TEST_COMPLETE. Search prints [SEARCH] only if the target is
missing. TEST_COMPLETE is deliberately not graded Task 4 SUCCESS.
The terminal command `stop` and browser E-Stop cancel queued autonomous actions.
Ctrl+C closes the worker and simulation. Do not issue manual motion during a test.

## Subsequent steps, after alignment is validated

Remove --task4-align-only to enable an initial slow approach. The bbox height
stopping thresholds (cup 0.32, bowl 0.11 of frame height) are STARTING ESTIMATES,
not calibrated values. Validate them on the Mac before recording any trial.
The code stops from visual size first, then evaluates planar distance and the
fresh matching detection after settling. It logs FAIL if the visual stop leaves
the robot farther than 0.80 m; it does not use this distance to drive closer.
Table/bench/wall contact aborts the mission. This is not general obstacle avoidance
and cannot guarantee contact-free motion before approach calibration.

`config/task4_objects.json` holds Task 2 object-center XY for evaluation only.
These are the uploaded scene's body XY coordinates; confirm they match the
agreed object centers in the final group scene. Changes to object positions
require updating evaluation configuration. Changes to camera FOV or object size
require visual-stop recalibration. The controller itself receives no object XY.

For graded natural-language commands, configure an existing Task 3 LLM provider
and run with --task4 instead of --task4-test. Both paths queue goto_object in the
same ordered motion deque. The direct `goto cup red` interface is DEVELOPMENT
ONLY and cannot replace the assignment's typed-English LLM demonstration.
The prior Task 3 parser/evaluation/history behavior has not been independently
validated here; this patch does not certify complete Task 3 compliance.

## Architecture and limitations

- Existing Task 2 front-camera buffer (nominal 15 Hz) -> one YOLO worker -> visual
  controller in the simulation thread -> existing velocity/policy/PD chain.
- YOLO inference is outside physics. The worker skips intermediate frames instead
  of building a backlog. Your measured YOLO times imply ~4-8 Hz; measure sustained
  rates with the simulation running and justify any departure from the suggested
  10-20 Hz. The first inference takes longer to initialize.
- Camera rendering currently depends on browser frame demand in the supplied
  runtime. Closing/hiding a browser can stop frames; stale results halt motion
  and then fail rather than steering on old frames.
- Color thresholds were checked on one stationary view. Motion, angle changes,
  background contamination and partial visibility still require evaluation.
- Search uses measured yaw, not elapsed time, to enforce a full-turn limit.
- Timeout: 60 seconds per mission. Missing detections hold still for 0.5 seconds
  before searching. Results older than 0.8 seconds immediately stop commands.
- Rendering remains the existing Task 2 implementation. No extra render context
  is created and runtime_control files are not replaced.
- Model loading/inference can contend for CPU despite using a worker thread.
- This is an integration checkpoint, not a finished assignment or demonstration.
  Remaining: Mac simulation verification, approach calibration, LLM setup and
  integration verification, ten navigation trials, failure analysis, video/report.

## Checks completed in the assistant workspace

Python syntax validation and seven pure-controller tests (steering signs, unique
frame confirmation, search completion, stale frames, timeout, fresh visual stop,
and immediate halt on target loss). Full MuJoCo/YOLO integration was NOT run here.

Run the controller tests from the project root:

    python tests/test_task4_navigation.py

All frame timestamps use monotonic wall time. ROS, retraining and new GPU setup
are not required. Preserve the original files when installing this checkpoint.
