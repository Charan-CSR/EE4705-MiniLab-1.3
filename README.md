# EE4705 MiniLab 1.3 — Command Your Robot Dog

LLM + YOLO powered quadruped project for EE4705 Human Robot Interaction.

## Project structure

- `src/platform/` — MuJoCo platform integration
- `src/camera/` — onboard RGB camera interface
- `src/motion_skills/` — timed move and closed-loop turn skills
- `src/llm/` — Task 3 structured-output LLM parser and chat interface
- `src/perception/` — YOLO detection, colour grounding and Task 4 search/approach
- `src/executor/` — command execution / integration
- `scenes/` — custom MJCF scene and object meshes
- `config/` — object positions and project configuration
- `tests/` — parser, motion and perception tests
- `scripts/` — runnable helper/demo scripts
- `results/` — evaluation outputs (CSV/logs)
- `docs/` — report-related material
- `videos/` — local demo videos (ignored by Git by default)

## Tasks

- Task 1 — LLM robot-control approaches and learned locomotion research
- Task 2 — platform, camera, scene and motion skills
- Task 3 — typed English commands via structured-output LLM parser
- Task 4 — YOLO object search and autonomous approach
- Task 5 — evaluation, report and demonstration videos

## Setup

The reference platform is `aoqianz/quadruped_mujoco`. Keep that upstream repository
separate or add it later as a Git submodule; do not copy its full source into this repository
unless the team intentionally decides to vendor it.

Detailed installation and run instructions will be added as implementation progresses.

## Security

Never commit API keys. Store secrets in environment variables or a local `.env` file.
`.env` is ignored by Git.
