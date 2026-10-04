# EE4705 MiniLab 1.3 — Command Your Robot Dog

**LLM + YOLO-Powered Quadruped | NUS EE4705 Human Robot Interaction**

This repository implements an integrated MuJoCo quadruped demonstration:

- **Task 2 — Platform and motion:** pre-trained quadruped locomotion, custom `yolo_lab` scene, front RGB camera, queued timed movement and feedback-controlled turns.
- **Task 3 — Language commands:** an LLM interprets typed English instructions and dispatches ordered motion actions, including multi-step commands.
- **Task 4 — Object search and approach:** the same command interface supports YOLO detections, color-based target selection, camera-guided searching, alignment, approach and success/failure reporting.

The implementation uses the existing learned walking policy and MuJoCo simulation. Gemini uses a cloud API; Ollama/Qwen can run locally. Camera perception and motion control run locally.

> **IMPORTANT — Ubuntu 22.04 on WSL2 (tested development setup):** The standard Task 4 `play.py --task4` launch sometimes terminated with **`Segmentation fault (core dumped)`** when MuJoCo and YOLO/PyTorch were initialized together. For the **full Task 3 + Task 4 integrated demo**, use the **verified library-preloading command in [Section 4](#4-full-end-to-end-demo-gemini--yolo-navigation-task-3--task-4)**. It starts the same `play.py` program; it is not a different implementation. The normal command may work on other machines. Keep the local browser panel open for camera frames.

## 1. Platform and prerequisites

- **Recommended OS:** Ubuntu 22.04 (native Linux or WSL2 on Windows). The integrated demo was run under Ubuntu 22.04 in WSL2.
- **Python:** Python 3.11 is the assignment's recommended environment. Confirm the exact interpreter version on the submission machine with `python --version`.
- **Environment manager:** Conda.
- **Dependencies:** see the repository's `requirements.txt`. Key packages include MuJoCo, ONNX Runtime, NumPy, Ultralytics/YOLO, PyTorch, TorchVision, and the configured Task 3 LLM client. The existing working WSL environment also imports `triton`.
- **Network:** required for Gemini and potentially first-time model or dependency downloads. Ollama can process commands locally once installed and its model is available.
- **Browser:** open the local simulator panel at `http://127.0.0.1:8765`.

### Clone and install

Run commands **from the repository root** (the directory containing `requirements.txt`, `src/`, and `quadruped_mujoco/`):

```
git clone https://github.com/Charan-CSR/EE4705-MiniLab-1.3.git
cd EE4705-MiniLab-1.3
conda create -n quadruped_mujoco python=3.11 -y
conda activate quadruped_mujoco
python -m pip install -r requirements.txt
python -m pip install -e ./quadruped_mujoco
```

> **Installation note:** The dependency-install commands above are the intended clean setup procedure, not a claim that they were verified on a fresh machine. If the nested `quadruped_mujoco` package is already installed, the editable install may be unnecessary. Validate setup in the final submitted ZIP on a clean environment.

## 2. Configure the LLM (Task 3 / integrated Task 4)

`src/llm/parser.py` supports two providers: **Gemini** (cloud) and **Ollama/Qwen3 4B** (local). The prompt instructs the LLM to return an ordered, structured list of `move`, `turn`, `goto_object`, `stop`, or `chat` actions. It uses default translation speeds of 0.4 (in the signed requested direction), requires explicit movement duration and turn angle, and returns an explanatory `chat` response when essential information is missing or a request cannot be interpreted. The `play.py` command handler rejects such invalid requests rather than queuing them.

### Gemini (tested for the integrated demo)

Create a local `.env` file **in the repository root**:

```dotenv
GEMINI_API_KEY=your_own_api_key_here
```

The parser loads this variable using `python-dotenv` and calls the configured Gemini model (`gemini-3.5-flash-lite` in the inspected source). Never commit the `.env` file or expose the actual key in screenshots, logs, the ZIP, or Git.

### Ollama / Qwen3 4B

Install and run Ollama, and make `qwen3:4b` available to the local service (for example, run `ollama pull qwen3:4b` after installing Ollama). This provider does not require a Gemini key. Ollama's setup and model availability must be verified on the target machine.

When the program starts, the terminal asks:

```text
Select LLM provider:
1 - Gemini
2 - Ollama (Qwen3 4B)
Enter choice [1/2]:
```

Choose **1** for Gemini or **2** for Ollama/Qwen3 4B. The latest integrated navigation test used **Gemini**.

## 3. Start the simulation

From the **repository root**, with Conda activated, the standard **Task 2 / Task 3** launch is:

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600
```

For **integrated Task 4**, use the preloading launch in Section 4 on the tested WSL2 system. The command above does **not** enable `--task4` object navigation. If your platform suffers a core dump during regular startup, use the same preload approach (without `--task4`) to run Task 2/3.

Open `http://127.0.0.1:8765` in your browser and **keep the page open**. Select the **`yolo_lab`** map in the browser control panel. The custom scene contains a **red cup**, **white cup**, and **green bowl**. Use the front onboard camera view to inspect the scene.

The simulator panel provides manual motion, reset, map switching and camera views. Task 2's `move(vx, vy, wz, duration)`, `turn(angle_deg)` and `stop()` motion APIs are integrated into the `play.py` control loop. The current submitted version does **not** include the former optional `--task2-demo` terminal interface; do not use that flag with the current `main` branch.

### Task 3 — Typed English commands and execution logs

After selecting an LLM provider, try these commands in the terminal, waiting for each accepted command to finish before entering another:

```text
Move forward for 2 seconds
Move forward for 2 seconds, then turn left 90 degrees
Move forward
```

The first tests a single movement; the second tests an ordered move and turn; and the third tests rejection because the duration is missing. Expected log categories:

- `[CMD] actions=… n=…` — accepted structured actions and their count.
- `[CMD] rejected reason=…` — missing information, unsupported command or parsing failure; rejected commands are not queued.
- `[LATENCY]` / `[PARSED]` — parse time and structured LLM output.
- `[QUEUE]` — accepted motion awaiting execution.
- `[EXEC]` — action starts; `[TURN]` — heading target and final angular error.
- `[DONE] move` / `[DONE] turn` — individual completion, followed by `[DONE] actions=… t=… s` **only once the whole command completes**.

An observed multi-step test produced `[DONE] actions=2 t=12.77 s`; the incomplete request produced `[CMD] rejected reason=Please specify the duration for moving forward.` without executing. Completion time measures elapsed command execution, not just LLM parse latency.

## 4. Full end-to-end demo: Gemini + YOLO navigation (Task 3 → Task 4)

**Recommended launch on the tested WSL2 development machine** — this is the command used successfully to run the combined `play.py` demonstration. The preload prevents the **observed** `Segmentation fault (core dumped)` during MuJoCo/YOLO/PyTorch startup:

```bash
python -c "import torch, torchvision, triton, mujoco; from ultralytics import YOLO; YOLO('yolo11n.pt'); import runpy, sys; sys.argv=['quadruped_mujoco/eg/play.py','--gui','--duration','3600','--task4']; runpy.run_path(sys.argv[0],run_name='__main__')"
```

**What the command does:** Imports PyTorch, TorchVision, Triton and MuJoCo; initializes YOLO with `yolo11n.pt`; then runs the actual repository `quadruped_mujoco/eg/play.py` with `--gui --duration 3600 --task4`. It uses the **same code and CLI flags** as a normal Task 4 launch, only changing library initialization order.

The equivalent, ordinary command is:

```bash
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4
```

On the development WSL2 machine, **prefer the longer command above**. On other machines, the ordinary command may work. The workaround is empirically verified on the development setup, not guaranteed on every environment. The first YOLO initialization may need access to model weights.

### Procedure for `Video_Task4` / final integrated demo

1. From the **repository root**, activate `quadruped_mujoco` and run the longer command above.
2. Open `http://127.0.0.1:8765` and keep the browser panel open. Select `yolo_lab` and allow the front RGB camera and YOLO detector to initialize.
3. At the terminal prompt, select **1 — Gemini** (with `GEMINI_API_KEY` configured) or **2 — Ollama/Qwen3 4B** (with the local service and model running).
4. Enter commands one at a time, waiting for each to finish:

   ```text
   Move forward for 2 seconds
   Move forward for 2 seconds, then turn left 90 degrees
   Move forward
   Go to the red cup.
   Go to the white cup.
   Go to the green bowl.
   ```

5. For object-search trials, reset or reposition between trials as appropriate. Include an initially invisible target and the **red-versus-white cup** same-class distinction.
6. Verify the logs: Task 3 `[CMD]`, `[EXEC]`, `[TURN]` and `[DONE]`; Task 4 `[SEARCH]` where needed, `[DETECT]`, `[FOUND]`, `[MISSION]` and command-level `[DONE]`.

The LLM generates `goto_object` actions specifying a target class and color. The controller searches by turning when necessary, centers the visual detection, approaches and stops. A successful mission requires a fresh matching detection and a measured planar distance of **≤ 0.80 m**. Ground-truth coordinates are used for **evaluation**, not steering.

## 5. Direct Task 4 testing (development only)

These modes **bypass the LLM** and are **not a substitute** for the Task 4 graded video.

**Alignment only** — rotates toward the target but intentionally does not approach:

```
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4-test --task4-align-only
```

**Full search and approach** — tests the same visual navigation controller without the LLM:

```
python quadruped_mujoco/eg/play.py --gui --duration 3600 --task4-test
```

In direct test mode, type the **object class before the color**:

```
goto cup red
goto cup white
goto bowl green

```

`goto white cup` and `goto green bowl` are **not** valid direct-test syntax. The expected alignment-only completion is `[MISSION] status=TEST_COMPLETE`, not graded navigation success. Full approach can report `[FOUND]` and `[MISSION] status=SUCCESS` if the success criteria are met.

For WSL2, if startup crashes, use the preloading command in Section 4 and change its `sys.argv` flags from `--task4` to `--task4-test`, optionally adding `--task4-align-only`. The preloaded `--task4` launch remains the recommended **final integrated** demonstration command on the tested development machine.

## 6. Safety and shutdown

- Do not manually drive the robot while an autonomous mission is running.
- Type `stop` at the test prompt or use the browser's **E-Stop** to cancel queued motion.
- Press **Ctrl+C** in the launching terminal to stop the simulation. A `KeyboardInterrupt` traceback after Ctrl+C is a manual shutdown, not evidence of mission failure.
- The controller includes a per-mission timeout (60 s in the Task 4 handover), stale-frame handling, and search termination. It is **not** a general obstacle-avoidance system.
- Keep the browser panel active to inspect the camera, even though later camera-rendering fixes aim to support frames without a browser viewer.

## 7. Scene, code, and results

| Path                                                     | Purpose                                                   |
| -------------------------------------------------------- | --------------------------------------------------------- |
| `quadruped_mujoco/eg/play.py`                            | Main simulator and integrated command/control entry point |
| `quadruped_mujoco/src/runtime_control/maps/yolo_lab.xml` | Registered custom scene/map                               |
| `src/perception/detector.py`                             | YOLO detection and color grounding                        |
| `src/perception/goto_object.py`                          | Visual search, alignment and approach controller          |
| `src/perception/trial_logging.py`                        | Task 4 trial logging                                      |
| `config/task4_objects.json`                              | Object centers used **only for evaluation**               |
| `tests/test_task4_navigation.py`                         | Pure Task 4 controller tests                              |
| `results/task4/trials/evaluation_02/`                    | Committed Task 4 evaluation data/screenshots              |
| `README_TASK4.md`                                        | Original Task 4 integration notes and limitations         |
| `CONTRIBUTIONS.md`                                       | Group contribution record                                 |
| `requirements.txt`                                       | Python dependencies                                       |

The `yolo_lab` scene uses a cup mesh scale of **0.02** and front camera field of view of **55°** as recorded in the integration notes. Changes to object size, position or camera FOV may require recalibrating visual stopping thresholds and evaluation coordinates.

Trial outputs may appear under `results/task4/trials/development/` or `results/task4/trials/fixcheck*/`. Alex’s later branch added development/fix-check results, while earlier local runs also generated **untracked** files. The contents of a fresh clone or submission ZIP depend on the actual committed files; do not assume every locally generated trial file is included.

## 8. Testing and evaluation

Task 4's pure-controller test entry point is:

```
python tests/test_task4_navigation.py
```

**Automated test history:** An earlier revision passed **6 of 7 tests**, with `test_wrong_color_searches_and_full_turn_fails` failing. Alex subsequently committed `d27063d` ("Fix outdated Task 4 sweep test (now 7/7 pass)"). Run the test command above on the submitted checkout to verify the final result; the older 6/7 figure should not be presented as a current test result. Separately, real-simulation trials successfully detected, aligned with, and approached objects; a recent integrated Gemini green-bowl trial logged `[FOUND] class=bowl color=green t=24.1 s d=0.626 m`, `[MISSION] status=SUCCESS reason=visual_stop_and_proximity_verified` and `[DONE] actions=1 t=24.22 s`. This single trial is not a substitute for the required multi-trial evaluation.

For the assignment report, evaluate **at least 10 Task 4 trials** with different starting poses and objects, including an initially invisible target and same-class/color disambiguation. Report detection accuracy, grounding accuracy, approach success rate, final planar distance and failure analysis. Task 3 requires a separate evaluation of **at least 20 utterances** using **two LLM services**, including paraphrases and invalid/out-of-scope commands; report accuracy, latency and cost. The tables and analysis belong in the **group report**; do not infer summary statistics from the mere presence of log files.

## 9. Demo video requirements

- **`Video_Task2`** — platform, custom scene, camera view and motion skills, including a visible `[TURN]` log.
- **`Video_Task3`** — typed English command with at least one multi-step motion instruction including a turn, plus a rejected command; show `[CMD] actions=… n=…`, `[CMD] rejected reason=…`, `[EXEC]` and final `[DONE] actions=… t=… s`.
- **`Video_Task4`** — use the **LLM-connected `--task4` mode**, not direct-test mode. Keep the terminal visible throughout. Show typed English object-search commands and `[CMD]`, `[SEARCH]`/`[DETECT]`, `[FOUND]`, `[MISSION]` in order for **at least two targets**, including one initially not visible and one same-class disambiguation. Red cup vs. white cup is the same-class pair; the green bowl provides a different-class demonstration.

The final ZIP must contain the **combined group report (PDF)**, **source code and install/run instructions**, **scene and mesh assets or download links**, **LLM prompt files**, and the **three demonstration videos**. Include group names/matriculation numbers, contribution attribution, citations and the required AI Usage Declaration. Follow the assignment's specified group ZIP naming convention.

## 10. Reproducibility notes and references

- Original simulation framework: [https://github.com/aoqianz/quadruped_mujoco](https://github.com/aoqianz/quadruped_mujoco)
- Assignment: *EE4705 MiniLab 1.3: Command Your Robot Dog: An LLM + YOLO Powered Quadruped*, AY 2026/27 Semester 1.
- This README describes the integrated behavior observed during development and the Task 4 handover. **Before submission**, verify Python/dependency versions, clean-install procedure, prompt-file or embedded-prompt requirements, contribution names and reproducibility from the final `main` checkout. No API secrets should be committed.
