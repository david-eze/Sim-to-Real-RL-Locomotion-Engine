# Continuous Control Reinforcement Learning for Autonomous Robotics

Production-ready, modular PyTorch & Gymnasium repository implementing continuous control Reinforcement Learning (Proximal Policy Optimization - PPO) with Multi-Objective Reward Function Engineering, Dynamic Curriculum Learning, and a Sim-to-Real Deployment Pipeline (ONNX & C++ Embedded Exporter).

Imagine trying to teach a baby robot how to walk. 

If I were to put a brand-new physical robot on the floor and tell it to figure out how to stand up on its own, it would crash, fall, and smash its expensive metal legs a thousand times. 

That’s where my **Sim-to-Real RL Locomotion Engine** comes in. Instead of breaking real hardware, I built a video-game-like virtual world on my computer to train an AI "brain" inside a physics simulation using trial and error. Once trained, I can upload that optimized brain directly into a physical robot.

---

## How the AI Thinks: Reinforcement Learning (RL)

Instead of hardcoding every single muscle or motor movement (like explicitly writing code to *"move leg A by 30 degrees, then move leg B"*), I used **Reinforcement Learning (RL)**.

![Reinforcement Learning Loop](./assets/reinforcement-learning.jpg)

Think of RL like training a dog with treats:
* **The Agent:** The robot’s AI brain.
* **The Environment:** The 3D physics simulator where gravity, friction, and momentum exist.
* **The Reward System:** I give the AI "points" whenever it takes a step forward without falling, and "penalties" whenever it falls over or consumes too much power.

At first, the robot flops around completely randomly. But after running millions of simulated attempts in a matter of seconds, the AI figures out balance, momentum, and smooth movement all on its own to maximize its score.

---

## What Does This Project Actually Do?

1. **Simulates the Physics:** I used MuJoCo to closely mimic real-world gravity, ground friction, and joint torque constraints.
2. **Trains the Neural Network:** I leveraged continuous control algorithms like **PPO** (Proximal Policy Optimization) to map raw sensor telemetry (such as balance tilt and joint angles) into smooth motor commands.
3. **Applies Domain Randomization:** To ensure the AI doesn't get "lazy" or overfitted to perfect computer conditions, I designed the simulation to randomly change floor slipperiness, apply unpredictable force pushes (virtual wind), and alter joint friction. This prepares the policy for the messy real world.
4. **Exports to Embedded Hardware:** I built an export pipeline that converts the finished PyTorch policy into optimized C++ code, making it lightweight enough to execute directly on a real-time embedded microcontroller.

---

## Project Structure

```
rl_robotics_biped/
├── config.py                 # Dataclasses for PPO hyperparameters, env, curriculum, and logging
├── envs/
│   ├── __init__.py
│   ├── bipedal_walker.py     # Custom Gymnasium Continuous Control Bipedal Environment (14D state, 4D action)
│   └── curriculum.py         # Dynamic Curriculum Learning Manager (scales push force, slope, penalties)
├── models/
│   ├── __init__.py
│   ├── actor_critic.py       # PyTorch Actor-Critic Networks (LayerNorm + Orthogonal Initialization)
│   └── ppo.py                # Custom Continuous PPO Engine (GAE, Clipped Surrogate, Value Loss, Entropy)
├── utils/
│   ├── __init__.py
│   └── logger.py             # Metrics Logger & TensorBoard Integration
├── export/
│   ├── __init__.py
│   └── exporter.py           # Sim-to-Real Exporter (PyTorch -> ONNX & zero-dependency C++ Header)
├── train.py                  # Main training entry point
├── evaluate.py               # Telemetry evaluator and real-time renderer script
└── README.md                 # Complete documentation & mathematical formulation
```

---

## Mathematical Formulation

### 1. Multi-Objective Reward Engineering

The continuous bipedal control agent is optimized under a multi-objective composite reward function balancing primary forward locomotion against physical efficiency and stability constraints:

$$R_t = w_{\text{fwd}} v_x - w_{\text{ctrl}} \alpha(t) \|a_t\|^2 - w_{\text{smooth}} \alpha(t) \|a_t - a_{t-1}\|^2 - w_{\text{posture}} \theta^2 + r_{\text{alive}} + R_{\text{fall}}$$

Where:
- $v_x$: Forward velocity ($m/s$).
- $\|a_t\|^2$: Energy consumption cost across joint actuators.
- $\|a_t - a_{t-1}\|^2$: Action smoothness penalty (penalizes high-frequency motor chattering).
- $\theta^2$: Torso pitch posture penalty (maintains upright orientation).
- $\alpha(t)$: Dynamic penalty scaling factor managed by the Curriculum Learning pipeline.
- $r_{\text{alive}}$: Constant positive reward per non-terminal step.
- $R_{\text{fall}}$: Terminal fall penalty ($-100.0$) if torso angle exceeds pitch threshold ($> 0.8\text{ rad}$).

---

### 2. Proximal Policy Optimization (PPO) Continuous Control Engine

#### Clipped Surrogate Policy Objective
To prevent destructive policy updates while exploring high-dimensional continuous torque spaces, PPO optimizes the clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where the probability ratio is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

#### Generalized Advantage Estimation
Advantages are estimated recursively using GAE to balance bias and variance:

$$\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) (1 - d_t) - V_\phi(s_t)$$

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

#### Value Function Loss with Value Clipping
The Critic state-value function $V_\phi(s)$ is trained via mean squared error over Monte-Carlo return targets $R_t = \hat{A}_t + V_\phi(s_t)$:

$$L^{\text{VF}}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \max \left( (V_\phi(s_t) - R_t)^2, \, (V_{\text{clip}}(s_t) - R_t)^2 \right) \right]$$

#### Entropy Regularization & Total Loss
An entropy bonus encourages policy exploration over continuous action distributions:

$$L^{\text{TOTAL}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\phi) - c_2 S[\pi_\theta]$$

---

## 3. Dynamic Curriculum Learning

The `CurriculumManager` dynamically modifies environment physics based on policy return performance:

| Stage | Name | Target Return | Push Disturbance ($\sigma$) | Slope | Penalty Scale $\alpha(t)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 0** | Flat Ground Walking | $\ge 150.0$ | $0.0\text{ N}$ | $0^\circ$ | $0.5$ |
| **Stage 1** | Rough Terrain & Smoothness | $\ge 230.0$ | $2.0\text{ N}$ | $2.0^\circ$ | $0.8$ |
| **Stage 2** | Robust Locomotion | $\ge 300.0$ | $5.0\text{ N}$ | $5.0^\circ$ | $1.0$ |

---

## Quickstart & Execution

### 1. Installation

Make sure PyTorch and Gymnasium are installed in your Python environment:

```bash
pip install torch numpy gymnasium tensorboard
```

### 2. Training the Agent

Launch PPO training with curriculum learning:

```bash
python train.py --timesteps 100000 --seed 42
```

### 3. Monitoring Training via TensorBoard

Launch TensorBoard to visualize reward convergence, policy entropy, value loss, and approximate KL divergence:

```bash
tensorboard --logdir ./logs/tb_logs
```

### 4. Evaluating Policy & Real-Time Telemetry

Run evaluation rollouts on a saved checkpoint:

```bash
python evaluate.py --model-path ./checkpoints/ppo_biped_final.pt --episodes 5 --render
```

---

## Sim-to-Real Embedded Hardware Deployment

Upon completing training, `train.py` automatically generates two hardware deployment artifacts under `./export_models/`:

1. **`policy.onnx`**: Standard ONNX model for deployment on ROS 2, NVIDIA Jetson, or PC nodes using ONNX Runtime or TensorRT.
2. **`embedded_policy.h`**: Standalone, zero-dependency C++ header file containing matrix operations and forward pass logic. Can be directly included in ARM Cortex / STM32 / ESP32 microcontroller firmwares for ultra-low-latency bare-metal motor control.

```cpp
#include "embedded_policy.h"

// Microcontroller real-time control loop
void control_loop() {
    float robot_state[14] = { /* IMU, joint encoders, foot contacts */ };
    float motor_torques[4];

    // Compute policy action (zero external library dependency)
    compute_robot_action(robot_state, motor_torques);

    // Send PWM / CAN torque commands to joint actuators
    actuate_motors(motor_torques);
}
```
