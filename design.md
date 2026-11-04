# Project: SO-ARM101 Simulation and VLA Control

## 1. Project Overview

I am working on a robotics project involving SO-ARM101 robotic arms, MuJoCo simulation, and Vision-Language-Action (VLA) models.

Our long-term goal is to develop a duck-like embodied robotic platform with:

- Two SO-ARM101 robotic arms mounted on the left and right sides.
- One or more head-mounted cameras providing an egocentric (first-person) view.
- A VLA model that receives visual observations, robot states, and natural language instructions and outputs robot actions.
- The ability to perform autonomous manipulation tasks, eventually including coordinated bimanual manipulation.

We currently have access to two physical SO-ARM-series arms, but I do not have the hardware with me right now. Their exact configuration (leader/follower pair versus two independently actuated follower arms) has not yet been verified.

**Therefore, the immediate goal is to develop and test the first two stages entirely in simulation, without requiring physical hardware.**

Do not assume that two physical follower arms are currently available.

We will use MuJoCo as the simulation environment, SO-ARM101 as the initial robot model, and Hugging Face LeRobot / SmolVLA for robot learning.

## 2. References and Official Resources

### SO-ARM101 Hardware and Simulation

Official SO-ARM100/101 GitHub repository:

https://github.com/TheRobotStudio/SO-ARM100

Official SO-101 simulation files (MJCF/URDF):

https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

SO-101 simulation README:

https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md

The simulation directory contains robot descriptions and MuJoCo scenes. Inspect the existing files and their include dependencies before implementing a loader.

Important details:

- The repository contains different calibration conventions, including `so101_new_calib.xml` and `so101_old_calib.xml`.
- The standard follower has five arm joints plus a gripper joint.
- The MJCF gripper convention does not necessarily match LeRobot's normalized gripper representation.
- A leader-arm URDF also exists, but the leader is not interchangeable with a powered follower.

### Hugging Face LeRobot

Official GitHub:

https://github.com/huggingface/lerobot

Documentation:

https://huggingface.co/docs/lerobot

SO-101 documentation:

https://huggingface.co/docs/lerobot/so101

### SmolVLA

Pretrained checkpoint:

https://huggingface.co/lerobot/smolvla_base

SmolVLA documentation:

https://huggingface.co/docs/lerobot/smolvla

SmolVLA is our preferred initial VLA because it is relatively lightweight and integrates with LeRobot.

We want to evaluate the pretrained checkpoint before collecting demonstrations or performing any fine-tuning.

### Existing MuJoCo + SmolVLA Integration

Strands Robots GitHub:

https://github.com/strands-labs/robots

First simulated robot tutorial:

https://strands-labs.github.io/robots/start/first-robot/

Pretrained SmolVLA simulation tutorial:

https://strands-labs.github.io/robots/start/first-policy/

These tutorials demonstrate SO-101 simulation and pretrained VLA inference.

Study these examples before implementing our own integration. Reuse working components where sensible rather than duplicating functionality.

However, do not confuse Strands agent-based natural-language control with actual VLA policy inference. We specifically want an image/state/language-conditioned action policy.

### MuJoCo

https://github.com/google-deepmind/mujoco

https://mujoco.readthedocs.io/

## 3. Development Environment

Primary development environment:

- Windows laptop
- NVIDIA RTX 4060 Laptop GPU (typically 8 GB VRAM)
- Python
- MuJoCo
- PyTorch
- Hugging Face LeRobot
- SmolVLA

We may later use a remote workstation or GPU cluster with NVIDIA RTX A6000 or RTX 6000 Ada (48 GB).

Requirements:

1. Prefer Python 3.12 unless dependency compatibility suggests otherwise.
2. Use `uv` for environment and dependency management where practical.
3. Provide Windows-compatible setup instructions.
4. Avoid assuming Linux-specific device paths or ROS dependencies.
5. Ensure Stage 1 works without CUDA.
6. Attempt Stage 2 locally if feasible, but support remote GPU inference later.
7. Pin compatible dependency versions and document any version constraints.
8. Do not require physical hardware for the current development stages.

## 4. Overall Project Roadmap

The project consists of three major stages.

### Stage 1 — Low-Level Robot Control

Develop a reliable, modular controller for SO-ARM101 inside MuJoCo.

### Stage 2 — Pretrained VLA Deployment

Integrate an existing pretrained VLA with MuJoCo to generate and execute robot actions without fine-tuning.

### Stage 3 — Data Collection and VLA Fine-tuning

Later, collect demonstrations, fine-tune pretrained policies, and evaluate improvements.

**For now, fully implement Stage 1 and Stage 2. Stage 3 should be documented and architecturally supported, but not implemented unless it is necessary for the earlier stages.**

---

# Stage 1: Low-Level Robot Control in MuJoCo

## Objective

Create a simulation environment in which SO-ARM101 can be controlled programmatically using Python.

The system should expose reusable robot-control interfaces that can eventually work with both simulated and physical robotic arms.

## Task 1.1 — Environment Setup

1. Inspect the existing project/repository.
2. Set up the Python environment.
3. Install MuJoCo and the required dependencies.
4. Obtain the official SO-101 MJCF and mesh assets.
5. Load the robot into a MuJoCo scene.
6. Launch a viewer and verify that the model loads correctly.

Use the official SO-101 model whenever possible.

Avoid recreating the URDF/MJCF from scratch.

Verify all asset paths, included XML files, joint definitions, and actuator definitions.

## Task 1.2 — Robot State and Actuator Inspection

Identify and document:

- Joint names and ordering
- Joint positions and velocities
- Actuator names and control ranges
- Joint limits
- Gripper configuration
- Initial robot pose
- Coordinate conventions
- Position units (radians versus degrees)
- Calibration conventions

Do not assume that `data.ctrl` is identical to the desired joint positions without checking the MJCF actuator configuration.

## Task 1.3 — Basic Joint Control

Implement:

- Reading joint positions
- Reading joint velocities
- Setting target joint positions
- Controlling individual joints
- Opening and closing the gripper
- Resetting the robot to a known pose
- Executing predefined joint trajectories
- Stepping the simulation

Suggested interface:

```python
class RobotArm:
    def reset(self): ...
    def get_joint_positions(self): ...
    def get_joint_velocities(self): ...
    def set_joint_targets(self, targets): ...
    def execute_trajectory(self, trajectory): ...
    def get_observation(self): ...
    def step(self): ...
    def render_camera(self, name): ...
```

This is an intended abstraction, not necessarily an existing LeRobot API.

Use actual MuJoCo actuators and a control loop rather than directly overwriting simulation state for normal motion.

## Task 1.4 — Control Safety and Validation

Implement:

- Joint limit checks
- Actuator control-range checks
- Maximum joint velocity constraints
- Reasonable acceleration limits if supported
- Safe reset behavior
- Action validation
- Error handling for invalid commands
- Tracking-error logging

Test individual joints before executing coordinated trajectories.

## Task 1.5 — Unified Robot Control Interface

Create a modular control API that separates the robot backend from the higher-level control policy.

The goal is to support:

```text
RobotControlInterface
    |
    +-- MuJoCoRobotController
    |
    +-- RealRobotController (future)
```

The real robot backend does not need to be implemented now.

The purpose is to avoid rewriting the entire software stack when physical hardware becomes available.

### Task 1.6 — Leader–Follower Teleoperation and Synchronization

**Objective:** Implement Leader–Follower synchronization as part of Stage 1, allowing a leader arm's joint movements to control a simulated or physical SO-101 follower arm.

This is a core requirement of the project, not merely a future data collection feature.

**Current development (without physical hardware):**

1. Implement a Virtual Leader controller that generates joint targets through an interactive UI, joint sliders, or recorded trajectories.
2. Connect the Virtual Leader to the MuJoCo SO-101 Follower.
3. Implement joint mapping, including joint ordering, calibration conventions, units, and gripper control.
4. Run a continuous teleoperation loop that sends Leader joint targets to the Follower.
5. Validate joint limits, velocity constraints, and safe initial alignment.
6. Measure tracking errors and simulated control latency.
7. Provide a runnable demonstration of Virtual Leader → Simulated Follower synchronization.

**Future physical hardware support:**

1. Integrate the Hugging Face LeRobot `SO101Leader` and `SO101Follower` interfaces.
2. Support Real Leader → Real Follower teleoperation.
3. Support Real Leader → Simulated Follower teleoperation for testing and demonstration collection.
4. Keep teleoperation independent from the VLA inference pipeline.
5. Allow the Follower to receive commands from either the teleoperation controller or the VLA policy, with only one active command source at a time.

**Suggested architecture:**

```text
                 Command Sources
                        |
          +-------------+-------------+
          |                           |
    Leader Controller            VLA Policy
    (Virtual / Real)             (Stage 2)
          |                           |
          +-------------+-------------+
                        |
                 Command Selector
                        |
                Joint Action Adapter
                        |
                 Safety Validation
                        |
                 Follower Controller
                 (MuJoCo / Real)
```

**Acceptance Criteria:**

- The Virtual Leader can control the simulated Follower.
- All arm joints and the gripper are correctly mapped.
- The Follower smoothly tracks Leader commands.
- Tracking errors and control latency are logged.
- The controller can switch between teleoperation and VLA control safely.
- The design supports future physical Leader–Follower synchronization without rewriting the simulation backend.

**Relevant official documentation:**

SO-101 hardware and calibration:
https://huggingface.co/docs/lerobot/so101

LeRobot teleoperation:
https://huggingface.co/docs/lerobot/il_robots

Official SO-101 simulation:
https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

Implement the simulated teleoperation functionality during Stage 1. Document physical hardware integration as future work.

## Stage 1 Acceptance Criteria

Stage 1 is complete when:

1. The official SO-101 model loads successfully in MuJoCo.
2. Joint positions and velocities can be retrieved.
3. Individual joints can be moved through Python.
4. The gripper can be opened and closed.
5. The robot can execute predefined trajectories.
6. All motion commands pass through validated actuator interfaces.
7. The controller API is reusable and independent of the VLA model.

Deliver a runnable demo that visibly moves the simulated robot.

---

# Stage 2: Pretrained VLA Deployment in Simulation

## Objective

Use a pretrained VLA model to control the SO-ARM101 inside MuJoCo, without collecting demonstrations or performing fine-tuning.

The initial model should be:

`lerobot/smolvla_base`

We want to establish the complete pipeline:

```text
MuJoCo Camera
      +
Robot Joint State
      +
Language Instruction
      |
      v
Pretrained SmolVLA
      |
      v
Predicted Action Chunk
      |
      v
Observation/Action Adapter
      |
      v
Safety Validation
      |
      v
MuJoCo Robot Controller
      |
      v
New Observation
      |
      +----> Repeat
```

## Task 2.1 — Camera Setup

Add at least one simulated RGB camera.

The camera should approximate the head-mounted egocentric camera planned for our future duck-like robot.

For now, the camera can be fixed relative to the simulated robot base or a virtual torso.

Requirements:

- Clear view of the workspace
- Visibility of the end effector and target object
- Configurable camera position and orientation
- Configurable resolution
- RGB image rendering
- Camera observation timestamps

Optional:

Add a wrist-mounted camera or an external overview camera for comparison.

The first implementation should work with the simplest feasible camera setup.

## Task 2.2 — Basic Manipulation Environment

Create a simple tabletop environment containing:

- One SO-101 follower arm
- A table or support surface
- A small graspable cube
- A target placement area
- One or more simulated cameras

Initial task:

"Pick up the cube and place it in the target area."

The environment should support:

- Resetting robot state
- Resetting object positions
- Rendering observations
- Detecting object contact
- Measuring grasp/placement success
- Logging trajectories

Start with one arm. Bimanual manipulation is a future extension.

## Task 2.3 — Load a Pretrained VLA

Load:

https://huggingface.co/lerobot/smolvla_base

Prefer the current official LeRobot model-loading and inference APIs.

Inspect and verify the checkpoint's expected:

- Camera inputs and image preprocessing
- Robot state representation
- Action representation
- Action chunk size
- Normalization requirements
- Input feature names
- Device and precision support

Do not assume that the pretrained model natively understands the SO-101 action space.

Use the Strands Robots integration as a reference if it simplifies this process.

Do not add any training or fine-tuning code to the Stage 2 execution path.

## Task 2.4 — Observation and Action Mapping

Develop an adapter between MuJoCo observations and SmolVLA.

Input mapping:

```text
MuJoCo RGB images
    -> VLA image features

MuJoCo joint positions
    -> VLA robot state

Natural language instruction
    -> VLA language input
```

Output mapping:

```text
VLA predicted actions
    -> Action decoding
    -> Unit conversion
    -> Joint ordering
    -> Joint and velocity limits
    -> MuJoCo actuator targets
```

Pay special attention to:

1. Joint ordering
2. Radians versus degrees
3. Gripper units and calibration
4. Model normalization/statistics
5. Action chunk timing
6. Control frequency
7. Camera input mapping
8. Padding or unused action dimensions

Use established model-specific adapters when available.

Do not treat arbitrary raw model output values as safe joint targets.

## Task 2.5 — Offline Inference

Before allowing the policy to move the robot:

1. Capture simulated RGB observations.
2. Read the robot state.
3. Provide a language instruction.
4. Run SmolVLA inference.
5. Print or visualize predicted actions.
6. Validate action dimensions and numerical values.
7. Check for NaNs, invalid actions, and out-of-range commands.

Save sample inputs and outputs for debugging.

This step should be runnable without executing predicted motions.

## Task 2.6 — Closed-Loop VLA Control

Implement a closed-loop policy execution system:

```python
while not done:
    observation = environment.get_observation()

    action_chunk = policy.predict(
        observation=observation,
        instruction=task_instruction,
    )

    validated_actions = action_adapter.validate(
        action_chunk
    )

    environment.execute(validated_actions)
```

This is conceptual pseudocode. Use the correct APIs from the installed libraries.

Requirements:

- Support action chunk execution.
- Re-observe the environment between chunks.
- Avoid uncontrolled actuator commands.
- Implement episode timeouts.
- Log predicted and executed actions.
- Record simulated camera frames.
- Keep physics simulation frequency separate from policy inference frequency.

## Task 2.7 — Zero-Shot Evaluation

Test several instructions:

- "Move the gripper toward the cube."
- "Pick up the cube."
- "Place the cube in the target area."

Evaluate three levels separately:

**Level 1: Inference**

Does the pretrained model successfully process the observations and generate valid action tensors?

**Level 2: Robot Control**

Can the predicted actions be mapped to the simulated SO-101 and executed safely?

**Level 3: Task Success**

Does the robot actually complete the requested manipulation task?

Do not assume that `smolvla_base` will succeed at Level 3.

The checkpoint may not have been trained on our exact camera setup, action representation, or simulation environment.

A successful execution pipeline is meaningful even if zero-shot task success is low.

If zero-shot performance is poor, document the failure modes rather than changing the model through training.

Optionally investigate a publicly available SO-101 task-specific checkpoint as an additional baseline, but keep it separate from the general pretrained SmolVLA experiment.

## Stage 2 Acceptance Criteria

Stage 2 is complete when:

1. The simulation provides valid camera observations.
2. SmolVLA can be loaded and used for inference.
3. Observation dimensions and feature names are compatible.
4. Predicted actions can be decoded and validated.
5. The simulated robot can execute predicted actions.
6. The process runs in a closed loop.
7. Logs and evaluation results are saved.
8. Zero-shot task success or failure is reported accurately.

Do not claim successful task completion unless it is actually observed and verified.

---

# Stage 3: Demonstration Collection and VLA Fine-tuning (Future Work)

## Objective

Improve manipulation performance using demonstrations and task-specific VLA fine-tuning.

This stage is not the immediate implementation priority.

### Planned Steps

1. Implement simulated teleoperation or an expert control policy.
2. Record demonstrations in MuJoCo.
3. Convert trajectories into LeRobotDataset format.
4. Later collect real-world demonstrations using the SO-101 Leader–Follower setup.
5. Fine-tune pretrained SmolVLA.
6. Compare zero-shot and fine-tuned policies.
7. Evaluate data efficiency using different demonstration counts.
8. Investigate simulation-to-real transfer.

Potential initial dataset sizes:

- 10 demonstrations for pipeline testing
- 50 demonstrations for an initial fine-tuning baseline
- 100–200 demonstrations for data-efficiency experiments

These are experimental starting points, not guaranteed sufficiency thresholds.

The eventual physical robot may use a different camera viewpoint and arm mounting configuration, so additional adaptation will likely be necessary.

---

# 5. Future Expansion: Duck-Like Bimanual Robot

Our final robot will have two robotic arms mounted on a larger body and a camera near the head/eyes.

Design the current software to support future extension to:

- Two independently actuated follower arms
- Joint state tracking for both arms
- Simultaneous left/right arm commands
- Coordinated bimanual actions
- Head-mounted egocentric cameras
- Additional wrist cameras
- Bimanual VLA action outputs
- Simulation-to-real deployment

Do not implement the entire bimanual system immediately.

However, avoid hard-coding assumptions that would make adding a second follower arm difficult.

The intended future VLA interface is approximately:

```text
Inputs:
    Head camera image
    Left arm state
    Right arm state
    Language instruction

Outputs:
    Left arm action chunk
    Right arm action chunk
```

If the physical arms turn out to be a leader/follower pair, the leader should primarily be used for teleoperation rather than automatically assumed to be a second powered follower.

---

# 6. Suggested Project Structure

You may adjust this structure based on the libraries and existing project, but keep the code modular.

```text
so101-vla/
|
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- simulation.yaml
|   |-- robot.yaml
|   |-- camera.yaml
|   |-- policy.yaml
|
|-- src/
|   |-- simulation/
|   |   |-- environment.py
|   |   |-- scene.py
|   |   |-- camera.py
|   |
|   |-- robots/
|   |   |-- base.py
|   |   |-- mujoco_so101.py
|   |
|   |-- control/
|   |   |-- joint_controller.py
|   |   |-- trajectory.py
|   |   |-- safety.py
|   |
|   |-- policies/
|   |   |-- smolvla.py
|   |   |-- observation_adapter.py
|   |   |-- action_adapter.py
|   |
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- rollout.py
|   |
|   |-- utils/
|       |-- logging.py
|
|-- scripts/
|   |-- launch_sim.py
|   |-- test_joint_control.py
|   |-- test_gripper.py
|   |-- test_vla_inference.py
|   |-- run_vla_sim.py
|
|-- tests/
|-- assets/
|-- outputs/
```

Avoid unnecessary abstractions and duplicate functionality already present in MuJoCo, LeRobot, or Strands Robots.

Prefer a minimal working implementation over a large framework.

---

# 7. Development Priorities

Please implement the project incrementally.

### Milestone 1

Run the SO-101 MuJoCo simulation.

### Milestone 2

Implement basic joint and gripper control.

### Milestone 3

Implement reusable RobotControlInterface and action validation.

### Milestone 4

Add a simulated camera and a tabletop manipulation task.

### Milestone 5

Load pretrained SmolVLA and perform offline inference.

### Milestone 6

Connect SmolVLA action outputs to MuJoCo control.

### Milestone 7

Run a closed-loop zero-shot policy rollout and evaluate the results.

Stage 3 is a future milestone and should be documented rather than fully implemented now.

---

# 8. Instructions for the Coding Agent

I want you to act as the robotics software engineer implementing this project.

**First:**

1. Inspect the current workspace.
2. Read the relevant official SO-101, MuJoCo, LeRobot, and SmolVLA documentation.
3. Review the existing Strands Robots simulation/VLA examples.
4. Identify the simplest reliable implementation strategy.
5. Create a concise implementation plan.

**Then:**

1. Implement Stage 1 completely.
2. Test Stage 1.
3. Implement Stage 2.
4. Test Stage 2.
5. Document known limitations and future improvements.

Do not stop after writing a high-level plan. Implement working code and run the available tests.

Use actual library APIs verified against the installed versions. Do not invent functions or assume undocumented behavior.

Reuse existing SO-101 models and available integration layers where practical.

Do not introduce fine-tuning into the first two stages.

Do not require hardware access.

Do not assume pretrained zero-shot task success.

Keep the final codebase easy to extend to two arms and physical hardware later.

## Expected Final Deliverables

1. Runnable SO-101 MuJoCo simulation.
2. Working Python joint/gripper control.
3. Modular robot control API.
4. Simulated camera observation pipeline.
5. SmolVLA pretrained inference integration.
6. VLA-to-MuJoCo action adapter.
7. Closed-loop simulated VLA rollout.
8. Tests and sample logs.
9. Clear Windows setup and execution instructions.
10. README explaining architecture, limitations, and the three-stage roadmap.

If dependencies, GPU memory, or model compatibility block any step, provide a concrete diagnosis and preserve a working earlier milestone.

**Start by inspecting the repository and official SO-101 simulation resources. Then implement Stage 1 before proceeding to Stage 2.**