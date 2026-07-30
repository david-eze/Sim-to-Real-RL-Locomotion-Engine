# Sim-to-Real RL Locomotion Engine

> A compact, reproducible PyTorch + Gymnasium prototype for learning continuous torque control in a reduced-order biped environment.

This project is designed to show the full control-software loop: define an observation/action interface, train a PPO actor–critic policy, increase task difficulty through a curriculum, inspect telemetry, and export the actor for deployment experiments.

**Important scope:** this repository uses a custom 14-state, reduced-order Gymnasium environment. It is **not** MuJoCo, contact-accurate physics, or hardware-validated sim-to-real transfer. Treat it as a clean control-learning prototype and a foundation for plugging in a higher-fidelity robot model.

## Demo

<img src="./assets/curriculum_rollout.gif" alt="Animated reduced-order biped rollout shown at flat-ground, rough-terrain, and disturbance curriculum stages." width="520" />

*A deterministic reference torque controller is used for these short visual rollouts so the documentation is repeatable even before PPO convergence. The trained-policy workflow is available separately through `train.py` and `evaluate.py`.*

<img src="./assets/curriculum_comparison.png" alt="Three biped poses labelled Stage 0 flat ground, Stage 1 rough terrain with two-newton push noise, and Stage 2 five-degree slope with five-newton push noise." width="760" />

## Engineering highlights

- **Continuous-control PPO:** Gaussian actor, value critic, clipped objective, entropy regularisation, GAE, gradient clipping, and linear learning-rate decay.
- **Explicit interface:** 14-dimensional proprioceptive observation and four continuous torque commands, with Gymnasium-compatible `reset`/`step` semantics.
- **Curriculum control:** flat ground → perturbations and roughness → slope and stronger disturbances; the stage manager can advance from evaluation return or be selected explicitly for tests.
- **Reproducible telemetry:** TensorBoard scalar logging, deterministic seeds, evaluation summaries, and four generated visual artifacts.
- **Deployment path:** ONNX actor export plus a dependency-free C++ header emitter that mirrors Linear, LayerNorm, and Tanh layers.

## System map

```text
14-state observation ──> ActorCriticPPO ──> 4 torque commands
         ^                                           │
         │                                           v
 curriculum parameters <── evaluation <── custom biped environment
                                      │
                                      └── TensorBoard / checkpoints / ONNX / C++ header
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py --timesteps 100000 --seed 42

python evaluate.py --model-path ./checkpoints/ppo_biped_final.pt --episodes 5

python simulate.py --all
```

## Generated simulation artifacts

| Artifact | What it demonstrates |
| --- | --- |
| `assets/curriculum_rollout.gif` | Animated reduced-order rollout through the three curriculum configurations. |
| `assets/curriculum_comparison.png` | Side-by-side curriculum conditions and their disturbance/slope settings. |
| `assets/gait_analysis.gif` | Animated hip–knee phase portrait for both legs under the reference controller. |
| `assets/performance_metrics.png` | Reward, forward speed, energy, and torso-drift telemetry from a short PPO smoke run. |

<img src="./assets/gait_analysis.gif" alt="Animated phase portrait tracing hip and knee angles for each leg under the reference torque controller." width="390" />

## Repository layout

```text
config.py
envs/
models/
train.py
evaluate.py
simulate.py
export/
assets/
```

## Validation and next steps

The current visual suite and imports were validated locally with:

```bash
python -m compileall config.py train.py evaluate.py simulate.py envs models export utils
python simulate.py --all
```

For a production sim-to-real project, the next engineering work should be to replace the reduced-order dynamics with a validated MuJoCo/Isaac/real-robot model, model actuator latency and saturation, implement terrain geometry rather than storing a roughness parameter, randomize physical parameters from measured ranges, and report held-out robustness trials with seeds and failure modes.

## License

MIT: see [LICENSE](./LICENSE).
