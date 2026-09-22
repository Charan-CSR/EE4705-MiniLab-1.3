# MuJoCo Runtime Control: Quick Porting and AI Adaptation Guide

`runtime_control` is a set of MuJoCo runtime components decoupled from the policy, robot model, and joint ordering. For a wheel-legged hybrid control adaptation, see `mujoco/w1w/play_gui.py`.

It provides a browser view, keyboard commands, map switching, onboard cameras, live PD parameters, motor delay, mass/friction/gravity adjustment, random pushes, randomization, reset, and emergency stop.

## 1. Files and Public Interface

```text
runtime_control/
├── pyproject.toml / setup.cfg  # Installable package metadata
├── MANIFEST.in                 # wheel/sdist resource declarations
├── src/runtime_control/
│   ├── __init__.py             # Stable public import entry point
│   ├── session.py              # Lifecycle wrapper for the composed scene
│   ├── adapter.py              # Robot/policy integration contract
│   ├── position_pd_adapter.py  # Generic implementation for pure position-PD robots
│   ├── resources.py            # API for bundled terrain resources
│   ├── integration.py          # Camera, viewer, and config factories
│   ├── runtime.py / panel.py   # Runtime and browser UI
│   ├── map_manager.py          # MJCF merging, map switching, and spawn points
│   ├── control.py              # Motor delay, PD, and torque limits
│   └── maps/                   # Terrain-only resources shipped with the wheel
└── eg/ / tests/                 # Runnable examples and regression tests
```

An editable install into the target Python environment is recommended:

```bash
python -m pip install -e /path/to/quadruped_mujoco
```

New adaptations should import from the package entry point rather than referencing internal files:

```python
from runtime_control import (
    ActionSpec, MapSpec, MotorCommandDelay, ParameterSpec,
    RuntimeControl, RuntimeScene, RobotAdapter, PositionPDAdapter,
    bundled_map_specs,
    compute_pd_torques, make_runtime_config,
    make_standard_robot_cameras, scale_torque_limits,
    setup_tracking_camera, standard_camera_options, viewer_context,
)```

The directory itself may be named `quadruped_mujoco`; after installation the stable Python package name is still
`runtime_control`.

The dependency direction is one-way: user projects import `runtime_control`, and the package never imports back into
W1W, Dog, ONNX, or any training framework. Observation construction, joint mapping, policy inference, and actuators
always stay in the user project.

## 2. Facts You Must Verify Before Porting

Do not guess the following from the robot's name. Confirm them from the target script, MJCF, training config, and the ONNX inputs/outputs:

1. The robot MJCF path and the name of the free-root base body;
2. MuJoCo's `qpos/qvel/ctrl` addresses and joint ordering;
3. The joint ordering, default angles, observation layout, and action scaling used by the policy;
4. The simulation timestep, control decimation, Kp/Kd, and the torque limit of each motor;
5. The root reset height, quaternion convention, and a safe spawn point for each map;
6. Whether the target script uses a "keyboard-controller class" or a global pressed-keys set;
7. Whether the actual runtime environment has `mujoco`, `Pillow`, `yaml`, and the policy backend.

The MuJoCo root quaternion is `wxyz`. The policy may use `xyzw`; the two must never be mixed.

### 2.1 12-DoF Pure Position PD

If every action is a joint target position, configure `PositionPDAdapter` directly:

```python
def build_obs(adapter, model, data, command, **context):
    joint_pos = data.qpos[adapter.qpos_adr]
    joint_vel = data.qvel[adapter.qvel_adr]
    # Build with the layout, scaling, and history ordering used in training; must match the policy.
    return make_policy_observation(joint_pos, joint_vel, command)

adapter = PositionPDAdapter(
    root_body_name="trunk",
    expected_dimensions=(19, 18, 12),
    joint_names=JOINT_NAMES,
    actuator_names=ACTUATOR_NAMES,
    default_positions=DEFAULT_POS,
    kp=KP,
    kd=KD,
    action_scale=0.25,
    torque_limits=TORQUE_LIMITS,
    observation_builder=build_obs,
)
```

### 2.2 16-DoF Hybrid Control

If actions include both positions and velocities, subclass `RobotAdapter` and implement
`bind/reset/build_observation/compute_control`. W1W's 12-leg position PD +
4-wheel velocity control is fully implemented in `mujoco/w1w/w1w_adapter.py`.

Both adapters validate the root body, `nq/nv/nu`, actuator IDs, output dimensions, and
NaN/Inf, so joint-ordering mistakes no longer enter the simulation silently.

## 3. Shortest Porting Workflow

### 3.1 Install and Import the Public Package

Use an editable install during development:

```bash
python -m pip install -e /path/to/quadruped_mujoco
```

For deployment to other projects, install the built wheel directly; there is no need to copy the source directory.
The source-path fallback in W1W is only for debugging inside this repository's parent tree; it is not a public API.

### 3.2 Declare Maps

```python
from runtime_control import bundled_map_specs

MAP_SPECS = bundled_map_specs(("race_track", "stairs"))
```

If a map XML ships with another robot, its root body must be excluded:

```python
from runtime_control import bundled_map_spec

MAP_SPECS["course"] = bundled_map_spec(
    "rc26_track", exclude_bodies=("old_robot_root",)
)
```

Maps are best kept as terrain-only MJCF containing only static terrain. Relative mesh, texture, and hfield paths are converted to absolute paths during merging.
Each map can set its own minimum half-thickness for thin plates via `minimum_box_half_thickness`,
and set contact parameters with `contact_params={"friction": "0.6 0.005 0.0001", "solref": "0.03 2"}`.
`strict=True` is the default: if an imported body contains a
joint/freejoint, an error is raised immediately, preventing maps from silently changing the robot's degrees of freedom.

### 3.3 Generate Genuinely Onboard Cameras

World-frame `track/trackcom` cameras are usually not much different from a third-person view. The public factory generates three fixed cameras genuinely attached under the robot's root body:

```python
ROBOT_CAMERAS = make_standard_robot_cameras(prefix="my_robot")
CAMERA_OPTIONS = standard_camera_options(prefix="my_robot")
```

The defaults assume a body frame of `+X` forward, `+Y` left, `+Z` up, with dimensions close to a mid-size quadruped. Adjust the local positions for different sizes:

```python
ROBOT_CAMERAS = make_standard_robot_cameras(
    prefix="my_robot",
    front_position=(0.20, 0.0, 0.12),
    rear_position=(-0.80, 0.0, 0.50),
    top_position=(0.0, 0.0, 1.00),
)
```

UI key names must exactly match the MJCF camera `name`. Using the same `prefix` for both functions avoids mismatches.

### 3.4 Merge the Robot and Maps

First build `runtime_config` per 3.5, then let the session object manage the temporary directory,
MJCF merging, `MjModel/MjData`, the runtime, and shutdown cleanup:

```python
scene = RuntimeScene.for_adapter(
    adapter,
    robot_xml=ROBOT_XML,
    map_specs=MAP_SPECS,
    runtime_config=runtime_config,
    robot_cameras=ROBOT_CAMERAS,
    dynamic_obstacle_map="dynamic_obstacles",
)
scene.open()
model, data, runtime = scene.model, scene.data, scene.runtime
```

All maps are compiled into a single model, and the runtime switches between them via geom groups without recreating `MjModel`.
`for_adapter()` automatically uses the adapter's root body/model dimensions, binding on open and
unbinding on close. Disabled maps have collisions turned off and are hidden.

### 3.5 Build the Runtime Config

The policy YAML can keep its original structure. Build the runtime config separately to avoid breaking training/inference parameters for the sake of the UI:

```python
map_labels = {"flat": "Flat ground", "stairs": "Stairs"}
map_spawns = {
    "flat": {"position": [0, 0, 0.45], "quaternion": [1, 0, 0, 0]},
    "stairs": {"position": [0, 0, 0.45], "quaternion": [1, 0, 0, 0]},
}

runtime_config = make_runtime_config(
    gui=args.gui,
    title="My Robot MuJoCo Live Tuning",
    maps=map_labels,
    map_spawns=map_spawns,
    kp=40.0,
    kd=1.0,
    torque_limit=35.0,
    initial_position=map_spawns["flat"]["position"],
    command=(1.0, 1.0, 1.0, 0.30),  # vx, vy, yaw, height
    height_range=(0.20, 0.40),
    cameras=CAMERA_OPTIONS,
    port=args.gui_port,
    randomization={
        "kp": [32.0, 48.0], "kd": [0.8, 1.2],
        "torque_limit": [28.0, 42.0],
        "motor_strength": [0.9, 1.1],
        "motor_delay_ms": [0.0, 15.0],
        "mass_scale": [0.9, 1.1], "payload_mass": [0.0, 2.0],
        "friction_scale": [0.5, 1.5], "gravity_z": [-10.3, -9.3],
    },
    parameters=[
        ParameterSpec(
            "wheel_velocity_kp", "Wheel velocity Kp", 0, 5, 0.05, 1.5
        ),
    ],
    actions=[
        ActionSpec("fall_side", "Fall-over test", shortcut="n", style="warn"),
    ],
    random_seed=1,
    snapshot_path="runtime_preset.json",
)
delay = MotorCommandDelay(model.opt.timestep)
```

`make_runtime_config` checks that every UI map has a spawn point, surfacing missing keys early.

### 3.6 The Browser and Native Viewer Must Be Mutually Exclusive

Do not unconditionally call `mujoco.viewer.launch_passive()` when `--gui` is set; otherwise the same simulation is rendered twice and the frame rate drops noticeably.

```python
display = scene.viewer(args.gui, key_callback=key_callback)
with display as viewer:
    if not args.gui:
        setup_tracking_camera(
            viewer, model, "base_link",
            distance=1.7, azimuth=135, elevation=-20,
        )

    while viewer.is_running():
        # simulation loop
        if not args.gui:
            viewer.sync()
```

With `--gui`, only a windowless loop context is provided and browser rendering is managed by `RuntimeControl`; the native viewer is created only without `--gui`.
Closing the browser panel notifies `viewer.is_running()` to exit and reclaims the HTTP
port and render thread.

### 3.7 Hook the Runtime Into Every Step

Global physical pressed-keys set version:

```python
if runtime.update_command(physical_pressed_keys):
    cmd = np.array([
        runtime_config["command"]["linear_x"],
        runtime_config["command"]["linear_y"],
        runtime_config["command"]["yaw"],
    ], dtype=np.float32)
else:
    cmd = original_get_commands()

state = runtime.runtime_control(model, data)
if runtime.consume_reset():
    reset_robot_from_runtime_config()
    delay.reset()
    runtime.reset_simulation_state()

runtime.apply_external_forces(model, data)
```

If a keyboard class already exists, use `RuntimeKeyboardMixin`. Do not forcibly refactor a simple global pressed-keys script into a class just for the mixin.

### 3.8 Apply Motor Parameters Before Writing `data.ctrl`

```python
applied_targets = delay.apply(targets, state["motor_delay_ms"])
data.ctrl[:] = compute_pd_torques(
    applied_targets, joint_positions, joint_velocities,
    state["kp"], state["kd"],
    motor_strength=state["motor_strength"],
    torque_limit=state["torque_limit"],
)
```

Kp, Kd, strength, and limits can be scalars or per-joint arrays.

If hip/thigh/calf have different nominal limits but the UI has only one slider, keep the original ratios:

```python
effective_limits = scale_torque_limits(
    nominal_limits, state["torque_limit"], reference_limit=35.55,
)
```

Then pass `effective_limits` to `compute_pd_torques`. Do not override all joint limits with a single scalar.

## 4. Full Reset Semantics

Map switching updates:

```python
runtime_config["simulation"]["initial_position"]
runtime_config["simulation"]["initial_quaternion"]
```

Reset must read these instead of continuing to hard-code positions. A full reset should also:

- Restore the root position, root quaternion, and default joint angles after `mj_resetData()`;
- Clear the policy's observation history and last action;
- Call `delay.reset()` to prevent stale pre-reset targets from being executed;
- Call `runtime.reset_simulation_state()` to clear residual external forces;
- Call `mj_forward()`.

## 5. Keyboard and Views

The browser panel already handles stuck keys caused by window blur, page hiding, the map/view dropdowns, emergency stop, and page close. The frontend sends release events, and the backend also clears keys as a fallback on map/camera switches. Adaptation scripts should not cache browser keys separately.

When physical function keys change the height or trigger the emergency stop, sync the panel:

```python
runtime_config["command"]["height"] = new_height
runtime.sync_height_to_panel()
runtime.sync_stop_to_panel()
```

## 6. Performance Principles

- The browser `fps` is a display target, not the physics simulation frequency;
- The plugin automatically enlarges the MuJoCo offscreen framebuffer according to `width/height`;
- Increasing `width/height` raises OpenGL, readback, and JPEG encoding costs;
- Prefer CSS upscaling; only raise the render resolution when the image is visibly blurry;
- Do not launch the native viewer when `--gui` is set;
- The multi-map model contains more geom metadata; that is the price of instant map switching;
- Prefer hfield for Perlin terrain rather than thousands of small boxes;
- `runtime_control()` is called every physics step; policy inference still runs at the original decimation.

## 7. Common Failures

### `ModuleNotFoundError: runtime_control`

Install this package into the current Python/conda environment per 3.1, and confirm the environment matches with
`python -m pip show mujoco-runtime-control`.

### Root body not found

`compose_scene(robot_body_name=...)` and `RuntimeControl(base_body_name=...)` must use the same real name.

### Robot disappears or falls after switching maps

Check that map's spawn height and `wxyz` quaternion; do not just tweak the global initial position.

### Few views, or "fixed cameras" have no effect

A world-frame `trackcom` is not an onboard camera. Use `make_standard_robot_cameras()` to inject them into the root body, and pass `standard_camera_options()` to the config.

### Motor delay has no effect

Confirm `delay.apply()` runs every physics step, not only on policy inference steps; call `delay.reset()` on reset.

### Motion breaks after adjusting torque

For differing motor limits, use `scale_torque_limits()` to preserve the ratios, and verify the direction of the policy/MuJoCo joint mapping.

### Port already in use

Expose `--gui-port` and pass it via `make_runtime_config(port=args.gui_port)`.

## 8. Acceptance Checklist

- [ ] `model.nq/nv/nu` match the original robot;
- [ ] The root body exists and the standard cameras' `cam_bodyid` equals the root body id;
- [ ] Every map has at least one `map_<name>_...` geom;
- [ ] Every map can be switched to and resets at a safe spawn point;
- [ ] Without `--gui`, only the native viewer opens;
- [ ] With `--gui`, only the browser opens;
- [ ] Switching views while holding a movement key does not cause stuck keys;
- [ ] Policy inputs, actions, and joint mapping are unchanged;
- [ ] With nominal settings, old and new PD outputs are identical;
- [ ] Kp/Kd, strength, delay, and torque actually affect `data.ctrl`;
- [ ] Reset clears the observation history, delay queue, and external forces;
- [ ] `scene.close()` is called on exit, or `with RuntimeScene(...)` is used.
- [ ] After closing, the same process can restart the panel on the same port.

Run the static checks first, then compile the composed MJCF in the actual environment:

```bash
python -m py_compile quadruped_mujoco/src/runtime_control/*.py <robot>/play.py
python -m unittest discover -s quadruped_mujoco/tests -v
```

## 9. Task Template Ready to Hand to an AI

```text
Please integrate mujoco/runtime_control into <target entry script>.

Requirements:
1. First read runtime_control/README.md, the target entry script, the policy config, and the robot MJCF;
2. Actually verify the root body, qpos/qvel/ctrl addresses, left/right joint ordering, and torque limits; do not guess;
3. Keep the existing observations, action scaling, default angles, and joint mapping;
4. Use MapSpec + compose_scene; do not copy runtime_control;
5. Use make_standard_robot_cameras + standard_camera_options;
6. Use make_runtime_config to build a standalone runtime config;
7. Use viewer_context, ensuring --gui opens only the browser and non---gui opens only the native viewer;
8. Hook RuntimeControl, MotorCommandDelay, external forces, and the real PD parameters into every physics step;
9. When per-joint torques differ, use scale_torque_limits to preserve the ratios;
10. On reset, read the current spawn point and clear the observation history, delay queue, and external forces;
11. Do not overwrite existing task-unrelated modifications in the target script;
12. Finally run py_compile, compile the composed MJCF, verify root body/camera ownership, map grouping, and
    old-vs-new PD numerical regression, and report the launch command.
```

## 10. Cross-Project Distribution

Stop copying the package's Python/XML/texture files. In the package directory run
`python -m pip wheel . --no-build-isolation --no-deps -w dist`,
then install `dist/mujoco_runtime_control-*.whl` in the user project. Terrain XML, textures, and
meshes are installed with the wheel and located by `bundled_map_specs()`. The user project keeps only
the robot MJCF, the policy adaptation code, spawn points, and its own terrain.
