# EE4705 MiniLab 1.3 — Command Your Robot Dog

**LLM + YOLO-Powered Quadruped | NUS EE4705 Human Robot Interaction**

This repository implements an integrated MuJoCo quadruped demonstration:

- **Task 2 — Platform and motion:** pre-trained quadruped locomotion, custom `yolo_lab` scene, front RGB camera, queued timed movement and feedback-controlled turns.
- **Task 3 — Language commands:** an LLM interprets typed English instructions and dispatches ordered motion actions, including multi-step commands.
- **Task 4 — Object search and approach:** the same command interface supports YOLO detections, color-based target selection, camera-guided searching, alignment, approach and success/failure reporting.

The implementation uses the existing learned walking policy and MuJoCo simulation. The LLM is cloud-based; camera perception and motion control run locally.

## 1. Platform and prerequisites

- **Recommended OS:** Ubuntu 22.04 (native Linux or WSL2 on Windows). The integrated demo was run under Ubuntu 22.04 in WSL2.
- **Python:** Python 3.11 is the assignment's recommended environment. Confirm the exact interpreter version on the submission machine with `python --version`.
- **Environment manager:** Conda.
- **Dependencies:** see the repository's `requirements.txt`. Key packages include MuJoCo, ONNX Runtime, NumPy, Ultralytics/YOLO, PyTorch, TorchVision, and the configured Task 3 LLM client. The existing working WSL environment also imports `triton`.
- **Network:** required for the chosen LLM API and possibly first-time model-weight download.
- **Browser:** open the local simulator panel at `http://127.0.0.1:8765`.

### Clone and install

Run commands **from the repository root** (the directory containing `requirements.txt`, `src/`, and `quadruped_mujoco/`):

```bash
git clone https://github.com/Charan-CSR/EE4705-MiniLab-1.3.git
cd EE4705-MiniLab-1.3
conda create -n quadruped_mujoco python=3.11 -y
conda activate quadruped_mujoco
python -m pip install -r requirements.txt
python -m pip install -e ./quadruped_mujoco
```

> **Installation note:** The dependency-install commands above are the intended clean setup procedure, not a claim that they were verified on a fresh machine. If the nested `quadruped_mujoco` package is already installed, the editable install may be unnecessary. Validate setup in the final submitted ZIP on a clean environment.

## 2. Configure the LLM (Task 3 / integrated Task 4)

The Task 3 parser was evaluated with **Gemini** and **Qwen**. To run the integrated object-search demo, select the existing, supported Task 3 provider (Gemini was used for the integrated test) and provide its API key using the configuration expected by the Task 3 implementation.

**Do not put API keys in source code, the README, screenshots, the ZIP, or Git.** Set the provider's required environment variable or use the project's existing local configuration mechanism. Consult the Task 3 parser/provider module for its **exact variable names and provider selection**; these details are implementation-specific and have not been independently verified from the available source snapshot.

To verify the key is available in the same shell, inspect only whether the relevant variable is set; do not print its value or commit it.

## 3. Start the simulation

From the **repository root**, with Conda activated:

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600
```

Open `http://127.0.0.1:8765` in your browser and **keep the page open**. Select the **`yolo_lab`** map in the browser control panel. The custom scene contains a **red cup**, **white cup**, and **green bowl**. Use the front onboard camera view to inspect the scene.

The simulator panel provides manual motion, reset, map switching and camera views. Task 2's queued movement and feedback-controlled turn interfaces are integrated into the `play.py` control loop. For Task 2's exact interactive command syntax and shortcuts, refer to the existing repository `README.md` and `play.py` help/interface; this guide does not assume unverified command names.

## 4. Full end-to-end demo: Gemini + YOLO navigation (Task 3 → Task 4)

**Use this mode for `Video_Task4` and the graded language-to-navigation demonstration.**

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4
```

1. Open `http://127.0.0.1:8765` and keep it visible.
2. Switch to `yolo_lab` and allow the front camera/YOLO detector to initialize.
3. At the **terminal's LLM command prompt**, type natural-language requests such as:

   ```text
   Go to the red cup.
   Go to the white cup.
   Go to the green bowl.
   ```

4. Let each mission finish before issuing the next one. Reset/reposition the robot between trials as needed, including a trial with a target initially outside the camera field of view.
5. Watch the terminal for the parsed `[CMD]`, `[SEARCH]` when necessary, `[DETECT]`, `[FOUND]` and `[MISSION]` logs.

Expected behavior: the LLM parses the English request into `goto_object(class, color)`, the camera/YOLO pipeline identifies the requested class and color, the robot rotates/searches if needed, aligns to the detected bounding box, approaches autonomously and stops. Success is only logged after a fresh matching detection and the measured planar distance is **≤ 0.80 m**. Ground-truth object coordinates are used for evaluation, **not for steering**.

### WSL2 startup workaround (verified on the development machine)

On the tested WSL2 setup, ordinary startup sometimes crashed with a native segmentation fault during concurrent MuJoCo rendering and YOLO/PyTorch initialization. **Preloading the libraries in the same Python process** allowed the simulation to run. If you encounter that problem, use this launch command instead:

```bash
python -c "import torch, torchvision, triton, mujoco; from ultralytics import YOLO; YOLO('yolo11n.pt'); import runpy, sys; sys.argv=['quadruped_mujoco/eg/play.py','--gui','--duration','3600','--task4']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

This workaround was observed to resolve the startup issue on the development machine; it is not a proven fix for every operating system or dependency version. The integrated Gemini/YOLO demo was reported to work after testing.

## 5. Direct Task 4 testing (development only)

These modes **bypass the LLM** and are **not a substitute** for the Task 4 graded video.

**Alignment only** — rotates toward the target but intentionally does not approach:

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4-test --task4-align-only
```

**Full search and approach** — tests the same visual navigation controller without the LLM:

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4-test
```

In direct test mode, type the **object class before the color**:

```text
goto cup red
goto cup white
goto bowl green
```

`goto white cup` and `goto green bowl` are **not** valid direct-test syntax. The expected alignment-only completion is `[MISSION] status=TEST_COMPLETE`, not graded navigation success. Full approach can report `[FOUND]` and `[MISSION] status=SUCCESS` if the success criteria are met.

For WSL2, if startup crashes, use the preloading command in Section 4 and change its `sys.argv` flags from `--task4` to `--task4-test`, optionally adding `--task4-align-only`.

## 6. Safety and shutdown

- Do not manually drive the robot while an autonomous mission is running.
- Type `stop` at the test prompt or use the browser's **E-Stop** to cancel queued motion.
- Press **Ctrl+C** in the launching terminal to stop the simulation. A `KeyboardInterrupt` traceback after Ctrl+C is a manual shutdown, not evidence of mission failure.
- The controller includes a per-mission timeout (60 s in the Task 4 handover), stale-frame handling, and search termination. It is **not** a general obstacle-avoidance system.
- Keep the browser panel active: in the current implementation, camera rendering may depend on browser frame demand.

## 7. Scene, code, and results

| Path | Purpose |
| --- | --- |
| `quadruped_mujoco/eg/play.py` | Main simulator and integrated command/control entry point |
| `quadruped_mujoco/src/runtime_control/maps/yolo_lab.xml` | Registered custom scene/map |
| `src/perception/detector.py` | YOLO detection and color grounding |
| `src/perception/goto_object.py` | Visual search, alignment and approach controller |
| `src/perception/trial_logging.py` | Task 4 trial logging |
| `config/task4_objects.json` | Object centers used **only for evaluation** |
| `tests/test_task4_navigation.py` | Pure Task 4 controller tests |
| `results/task4/trials/evaluation_02/` | Committed Task 4 evaluation data/screenshots |
| `README_TASK4.md` | Original Task 4 integration notes and limitations |
| `CONTRIBUTIONS.md` | Group contribution record |
| `requirements.txt` | Python dependencies |

The `yolo_lab` scene uses a cup mesh scale of **0.02** and front camera field of view of **55°** as recorded in the integration notes. Changes to object size, position or camera FOV may require recalibrating visual stopping thresholds and evaluation coordinates.

Additional local trial results may appear under `results/task4/trials/development/`. This folder was **untracked** when the merged branch was pushed, so it may not be included in a fresh Git clone or submitted ZIP unless added explicitly.

## 8. Testing and evaluation

Task 4's pure-controller test entry point is:

```bash
python tests/test_task4_navigation.py
```

**Known test issue:** In the earlier observed test run, **6 of 7 tests passed**, while `test_wrong_color_searches_and_full_turn_fails` failed because the simulated full-turn test did not set `nav.outcome` to `FAIL` within the test's sampled sequence. This result must be rechecked before claiming a fully passing automated test suite. Separately, real-simulation trials successfully detected, aligned with, and approached objects; one observed green-bowl trial logged `d=0.600 m` and `status=SUCCESS`.

For the assignment report, evaluate **at least 10 Task 4 trials** with different starting poses and objects, including an initially invisible target and same-class/color disambiguation. Report detection accuracy, grounding accuracy, approach success rate, final planar distance and failure analysis. Task 3 requires a separate evaluation of **at least 20 utterances** using **two LLM services**, including paraphrases and invalid/out-of-scope commands; report accuracy, latency and cost. The tables and analysis belong in the **group report**; do not infer summary statistics from the mere presence of log files.

## 9. Demo video requirements

- **`Video_Task2`** — platform, custom scene, camera view and motion skills, including a visible `[TURN]` log.
- **`Video_Task3`** — typed English command with at least one multi-step motion instruction including a turn, plus a rejected command; show `[CMD]`, `[EXEC]` and `[DONE]`.
- **`Video_Task4`** — use the **LLM-connected `--task4` mode**, not direct-test mode. Keep the terminal visible throughout. Show typed English object-search commands and `[CMD]`, `[SEARCH]`/`[DETECT]`, `[FOUND]`, `[MISSION]` in order for **at least two targets**, including one initially not visible and one same-class disambiguation. Red cup vs. white cup is the same-class pair; the green bowl provides a different-class demonstration.

The final ZIP must contain the **combined group report (PDF)**, **source code and install/run instructions**, **scene and mesh assets or download links**, **LLM prompt files**, and the **three demonstration videos**. Include group names/matriculation numbers, contribution attribution, citations and the required AI Usage Declaration. Follow the assignment's specified group ZIP naming convention.

## 10. Reproducibility notes and references

- Original simulation framework: https://github.com/aoqianz/quadruped_mujoco
- Assignment: *EE4705 MiniLab 1.3: Command Your Robot Dog: An LLM + YOLO Powered Quadruped*, AY 2026/27 Semester 1.
- This README describes the integrated behavior observed during development and the Task 4 handover. **Before submission**, verify the exact Task 3 provider configuration, Python/dependency versions, clean-install procedure, paths for prompt files, contribution names and reproducibility from the final `main` checkout. No API secrets should be committed.
