import os, sys, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EnvConfig, PPOConfig, CurriculumConfig
from envs import BipedalWalkerCustomEnv, CurriculumManager
from models import ActorCriticPPO


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT_DIR, exist_ok=True)


COLOURS = {
    "bg":       "#0F1923",
    "primary":  "#00E5FF",
    "secondary":"#FF6B35",
    "accent":   "#76FF03",
    "warn":     "#FFD600",
    "text":     "#E0E0E0",
    "grid":     "#1E2D3D",
    "stage0":   "#4FC3F7",
    "stage1":   "#FFB74D",
    "stage2":   "#81C784",
}

plt.rcParams.update({
    "figure.facecolor": COLOURS["bg"],
    "axes.facecolor":   COLOURS["bg"],
    "axes.edgecolor":   COLOURS["grid"],
    "axes.labelcolor":  COLOURS["text"],
    "text.color":       COLOURS["text"],
    "xtick.color":      COLOURS["text"],
    "ytick.color":      COLOURS["text"],
    "grid.color":       COLOURS["grid"],
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "font.size":        11,
})


def draw_robot(ax, state, colour=COLOURS["primary"], alpha=1.0):
    hull_angle = state[0]
    hip1, knee1 = state[4], state[6]
    hip2, knee2 = state[8], state[10]


    torso_len = 0.8
    torso_cx, torso_cy = 0.0, 0.95
    cos_a, sin_a = np.cos(hull_angle), np.sin(hull_angle)
    corners = np.array([
        [-torso_len/2, -0.15],
        [ torso_len/2, -0.15],
        [ torso_len/2,  0.15],
        [-torso_len/2,  0.15],
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners @ rot.T + np.array([torso_cx, torso_cy])
    torso = plt.Polygon(corners, color=colour, alpha=alpha, ec="white", lw=0.5, zorder=3)
    ax.add_patch(torso)


    hip1_x = torso_cx - 0.25 * cos_a
    hip1_y = torso_cy - 0.25 * sin_a
    hip2_x = torso_cx + 0.25 * cos_a
    hip2_y = torso_cy + 0.25 * sin_a


    thigh1_x = hip1_x + 0.35 * np.sin(hip1)
    thigh1_y = hip1_y - 0.35 * np.cos(hip1)
    ax.plot([hip1_x, thigh1_x], [hip1_y, thigh1_y], color=colour, lw=4, alpha=alpha, zorder=2)
    shin1_x = thigh1_x + 0.35 * np.sin(hip1 + knee1)
    shin1_y = thigh1_y - 0.35 * np.cos(hip1 + knee1)
    ax.plot([thigh1_x, shin1_x], [thigh1_y, shin1_y], color=colour, lw=3, alpha=alpha, zorder=2)
    ax.scatter([shin1_x], [shin1_y], s=40, c=colour, alpha=alpha, zorder=4)


    thigh2_x = hip2_x + 0.35 * np.sin(hip2)
    thigh2_y = hip2_y - 0.35 * np.cos(hip2)
    ax.plot([hip2_x, thigh2_x], [hip2_y, thigh2_y], color=colour, lw=4, alpha=alpha, zorder=2)
    shin2_x = thigh2_x + 0.35 * np.sin(hip2 + knee2)
    shin2_y = thigh2_y - 0.35 * np.cos(hip2 + knee2)
    ax.plot([thigh2_x, shin2_x], [thigh2_y, shin2_y], color=colour, lw=3, alpha=alpha, zorder=2)
    ax.scatter([shin2_x], [shin2_y], s=40, c=colour, alpha=alpha, zorder=4)


    for (jx, jy) in [(hip1_x, hip1_y), (hip2_x, hip2_y),
                      (thigh1_x, thigh1_y), (thigh2_x, thigh2_y)]:
        ax.scatter([jx], [jy], s=15, c="white", alpha=alpha, zorder=5)


def reference_torque_controller(step: int) -> np.ndarray:
    phase = step * 0.16
    return np.array([0.60, 0.55 * np.sin(phase), 0.60, 0.55 * np.cos(phase)], dtype=np.float32)


def simulate_training_progress():
    print("[simulate] Generating curriculum-rollout GIF …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    ppo_config = PPOConfig(total_timesteps=12_000, hidden_dims=(64, 64), n_steps=512,
                           batch_size=128, n_epochs=3)
    curriculum_config = CurriculumConfig()

    env = BipedalWalkerCustomEnv(config=env_config)
    policy = ActorCriticPPO(state_dim=env_config.state_dim,
                            action_dim=env_config.action_dim,
                            config=ppo_config).to(device)
    curriculum_mgr = CurriculumManager(config=curriculum_config)
    curriculum_mgr.apply_stage_to_env(env)


    from models import PPOTrainer
    trainer = PPOTrainer(policy=policy, env_config=env_config,
                         ppo_config=ppo_config, device=device)

    state, _ = env.reset()
    stage_snapshots = {0: [], 1: [], 2: []}
    episode_return = 0.0
    best_return = -1e9

    for step in range(1, ppo_config.total_timesteps + 1):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, val = policy.step(state_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        trainer.buffer.add(state, action_np, reward, val.item(), log_prob.item(), done)
        state = next_state
        episode_return += reward

        if done:
            state, _ = env.reset()
            episode_return = 0.0

        if step % ppo_config.n_steps == 0:
            with torch.no_grad():
                last_val = policy.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            trainer.buffer.compute_gae(last_val, done, ppo_config.gamma, ppo_config.gae_lambda)
            progress = 1.0 - (step / ppo_config.total_timesteps)
            trainer.update_learning_rate(progress)
            trainer.train_step()


        capture_stages = {4_000: 0, 8_000: 1, 12_000: 2}
        if step in capture_stages:
            stage_idx = capture_stages[step]
            curriculum_mgr.set_stage(stage_idx)
            curriculum_mgr.apply_stage_to_env(env)
            visual_state, _ = env.reset(seed=100 + stage_idx)
            for visual_step in range(30):
                stage_snapshots[stage_idx].append(visual_state.copy())
                visual_state, _, terminated, truncated, _ = env.step(reference_torque_controller(visual_step))
                if terminated or truncated:
                    visual_state, _ = env.reset()
            curriculum_mgr.set_stage(0)
            curriculum_mgr.apply_stage_to_env(env)
            state, _ = env.reset()


    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
    stage_names = ["Stage 0 — Flat Ground", "Stage 1 — Rough Terrain", "Stage 2 — Disturbances"]
    stage_cols = [COLOURS["stage0"], COLOURS["stage1"], COLOURS["stage2"]]

    def init_anim():
        for ax in axes:
            ax.clear()
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.8)
            ax.set_aspect("equal")
            ax.axis("off")
        return axes

    def update_anim(frame):
        for row, (sid, ax) in enumerate(zip(stage_snapshots, axes)):
            ax.clear()
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.8)
            ax.set_aspect("equal")
            ax.axis("off")

            ax.axhline(0, color=COLOURS["grid"], lw=2)

            ax.set_title(stage_names[row], color=stage_cols[row],
                         fontsize=10, fontweight="bold", pad=4)

            frames = stage_snapshots[sid]
            if frames:
                idx = min(frame, len(frames) - 1)
                draw_robot(ax, frames[idx], colour=stage_cols[row])
        return axes

    anim = FuncAnimation(fig, update_anim, frames=30, init_func=init_anim,
                         blit=False, interval=120)
    path = os.path.join(OUT_DIR, "curriculum_rollout.gif")
    anim.save(path, writer=PillowWriter(fps=8), dpi=120)
    plt.close(fig)
    print(f"  ✓ Saved {path}")
    return path


def simulate_curriculum_comparison():
    print("[simulate] Generating curriculum-comparison image …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    ppo_config = PPOConfig(total_timesteps=8_000, hidden_dims=(64, 64), n_steps=512,
                           batch_size=128, n_epochs=3)
    curriculum_config = CurriculumConfig()

    env = BipedalWalkerCustomEnv(config=env_config)
    policy = ActorCriticPPO(state_dim=env_config.state_dim,
                            action_dim=env_config.action_dim,
                            config=ppo_config).to(device)
    curriculum_mgr = CurriculumManager(config=curriculum_config)


    from models import PPOTrainer
    trainer = PPOTrainer(policy=policy, env_config=env_config,
                         ppo_config=ppo_config, device=device)
    state, _ = env.reset()
    for step in range(1, ppo_config.total_timesteps + 1):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, val = policy.step(state_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        trainer.buffer.add(state, action_np, reward, val.item(), log_prob.item(), done)
        state = next_state
        if done:
            state, _ = env.reset()
        if step % ppo_config.n_steps == 0:
            with torch.no_grad():
                last_val = policy.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            trainer.buffer.compute_gae(last_val, done, ppo_config.gamma, ppo_config.gae_lambda)
            progress = 1.0 - (step / ppo_config.total_timesteps)
            trainer.update_learning_rate(progress)
            trainer.train_step()
        if step % 3000 == 0:
            eval_return = _evaluate(policy, env, device, 3)
            curriculum_mgr.update_curriculum(eval_return, env)


    frames = {}
    for stage_idx in range(3):
        curriculum_mgr.set_stage(stage_idx)
        curriculum_mgr.apply_stage_to_env(env)
        state, _ = env.reset()

        for rollout_step in range(20):
            state, _, _, _, _ = env.step(reference_torque_controller(rollout_step))
        frames[stage_idx] = state.copy()


    stage_info = [
        ("Flat Ground", "No disturbances\n0° slope", COLOURS["stage0"]),
        ("Rough Terrain", "Push σ=2.0 N\n2° slope", COLOURS["stage1"]),
        ("Robust Locomotion", "Push σ=5.0 N\n5° slope", COLOURS["stage2"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for idx, (title, desc, colour) in enumerate(stage_info):
        ax = axes[idx]
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.axhline(0, color=COLOURS["grid"], lw=2)

        badge = FancyBboxPatch((-0.9, 1.45), 1.8, 0.25,
                               boxstyle="round,pad=0.05",
                               facecolor=colour, edgecolor="white", lw=0.5,
                               alpha=0.25)
        ax.add_patch(badge)
        ax.text(0, 1.57, f"Stage {idx}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=colour)
        ax.text(0, 1.30, desc, ha="center", va="center",
                fontsize=8, color=COLOURS["text"], alpha=0.8)
        if idx in frames:
            draw_robot(ax, frames[idx], colour=colour)
    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, "curriculum_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLOURS["bg"])
    plt.close(fig)
    print(f"  ✓ Saved {path}")
    return path


def simulate_performance_metrics():
    print("[simulate] Generating performance-metrics chart …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    ppo_config = PPOConfig(total_timesteps=12_000, hidden_dims=(64, 64), n_steps=512,
                           batch_size=128, n_epochs=3)
    curriculum_config = CurriculumConfig()

    env = BipedalWalkerCustomEnv(config=env_config)
    policy = ActorCriticPPO(state_dim=env_config.state_dim,
                            action_dim=env_config.action_dim,
                            config=ppo_config).to(device)
    curriculum_mgr = CurriculumManager(config=curriculum_config)
    curriculum_mgr.apply_stage_to_env(env)

    from models import PPOTrainer
    trainer = PPOTrainer(policy=policy, env_config=env_config,
                         ppo_config=ppo_config, device=device)


    steps_log, reward_log = [], []
    speed_log, energy_log, drift_log = [], [], []
    stage_log = []

    state, _ = env.reset()
    ep_return = 0.0
    ep_speeds, ep_energies, ep_drifts = [], [], []

    for step in range(1, ppo_config.total_timesteps + 1):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, val = policy.step(state_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        trainer.buffer.add(state, action_np, reward, val.item(), log_prob.item(), done)
        state = next_state
        ep_return += reward
        ep_speeds.append(info["vel_x"])
        ep_energies.append(info["energy_cost"])
        ep_drifts.append(abs(info["hull_angle"]))

        if done:
            state, _ = env.reset()
            ep_return = 0.0
            ep_speeds, ep_energies, ep_drifts = [], [], []

        if step % ppo_config.n_steps == 0:
            with torch.no_grad():
                last_val = policy.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            trainer.buffer.compute_gae(last_val, done, ppo_config.gamma, ppo_config.gae_lambda)
            progress = 1.0 - (step / ppo_config.total_timesteps)
            trainer.update_learning_rate(progress)
            trainer.train_step()

        if step % 1000 == 0:
            eval_return = _evaluate(policy, env, device, 3)
            steps_log.append(step)
            reward_log.append(eval_return)
            stage_log.append(curriculum_mgr.current_stage_idx)

            spd, eng, drf = _eval_metrics(policy, env, device, 2)
            speed_log.append(spd)
            energy_log.append(eng)
            drift_log.append(drf)
            curriculum_mgr.update_curriculum(eval_return, env)


    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = ["Mean Episode Return", "Forward Speed (m/s)",
              "Energy per Step", "Torso Drift (rad)"]
    data = [reward_log, speed_log, energy_log, drift_log]
    colours_plot = [COLOURS["primary"], COLOURS["accent"],
                    COLOURS["warn"], COLOURS["secondary"]]

    for ax, title, vals, clr in zip(axes.flat, titles, data, colours_plot):
        ax.plot(steps_log, vals, color=clr, lw=1.5, marker=".", markersize=3)
        ax.fill_between(steps_log, vals, alpha=0.1, color=clr)
        ax.set_title(title, fontsize=11, fontweight="bold", color=clr)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.grid(True, alpha=0.3)

        for sid in range(3):
            mask = [s == sid for s in stage_log]
            if any(mask):
                x_vals = [steps_log[i] for i, m in enumerate(mask) if m]
                y_vals = [vals[i] for i, m in enumerate(mask) if m]
                if len(x_vals) > 1:
                    ax.fill_between(x_vals, 0, 1, alpha=0.04,
                                    color=[COLOURS["stage0"],
                                           COLOURS["stage1"],
                                           COLOURS["stage2"]][sid],
                                    transform=ax.get_xaxis_transform(),
                                    label=f"Stage {sid}" if ax == axes[0,0] else "")

    axes[0,0].legend(fontsize=7, loc="lower right",
                     facecolor=COLOURS["bg"], edgecolor=COLOURS["grid"])
    plt.tight_layout(pad=1.5)
    path = os.path.join(OUT_DIR, "performance_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLOURS["bg"])
    plt.close(fig)
    print(f"  ✓ Saved {path}")
    return path


def simulate_gait_analysis():
    print("[simulate] Generating gait-analysis GIF …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = EnvConfig()
    ppo_config = PPOConfig(total_timesteps=8_000, hidden_dims=(64, 64), n_steps=512,
                           batch_size=128, n_epochs=3)

    env = BipedalWalkerCustomEnv(config=env_config)
    policy = ActorCriticPPO(state_dim=env_config.state_dim,
                            action_dim=env_config.action_dim,
                            config=ppo_config).to(device)

    from models import PPOTrainer
    trainer = PPOTrainer(policy=policy, env_config=env_config,
                         ppo_config=ppo_config, device=device)
    state, _ = env.reset()
    for step in range(1, ppo_config.total_timesteps + 1):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, val = policy.step(state_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        trainer.buffer.add(state, action_np, reward, val.item(), log_prob.item(), done)
        state = next_state
        if done:
            state, _ = env.reset()
        if step % ppo_config.n_steps == 0:
            with torch.no_grad():
                last_val = policy.critic(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            trainer.buffer.compute_gae(last_val, done, ppo_config.gamma, ppo_config.gae_lambda)
            progress = 1.0 - (step / ppo_config.total_timesteps)
            trainer.update_learning_rate(progress)
            trainer.train_step()


    state, _ = env.reset()
    hip1_hist, knee1_hist = [], []
    hip2_hist, knee2_hist = [], []
    for rollout_step in range(200):
        state, _, terminated, truncated, _ = env.step(reference_torque_controller(rollout_step))
        hip1_hist.append(state[4])
        knee1_hist.append(state[6])
        hip2_hist.append(state[8])
        knee2_hist.append(state[10])
        if terminated or truncated:
            break


    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Hip Angle (rad)", fontsize=10)
    ax.set_ylabel("Knee Angle (rad)", fontsize=10)
    ax.set_title("Reference-Controller Joint Phase Portrait", fontsize=11,
                 fontweight="bold", color=COLOURS["primary"])
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color=COLOURS["grid"], lw=0.5)
    ax.axvline(0, color=COLOURS["grid"], lw=0.5)

    line1, = ax.plot([], [], color=COLOURS["stage0"], lw=2, alpha=0.8, label="Leg 1")
    line2, = ax.plot([], [], color=COLOURS["stage1"], lw=2, alpha=0.8, label="Leg 2")
    scatter1 = ax.scatter([], [], s=30, c=COLOURS["stage0"], alpha=0.6, zorder=5)
    scatter2 = ax.scatter([], [], s=30, c=COLOURS["stage1"], alpha=0.6, zorder=5)
    ax.legend(fontsize=8, facecolor=COLOURS["bg"], edgecolor=COLOURS["grid"])

    n_frames = min(60, len(hip1_hist))

    def init_gait():
        line1.set_data([], [])
        line2.set_data([], [])
        scatter1.set_offsets(np.empty((0, 2)))
        scatter2.set_offsets(np.empty((0, 2)))
        return line1, line2, scatter1, scatter2

    def update_gait(frame):
        end = int((frame + 1) / n_frames * len(hip1_hist))
        line1.set_data(hip1_hist[:end], knee1_hist[:end])
        line2.set_data(hip2_hist[:end], knee2_hist[:end])
        if end > 0:
            scatter1.set_offsets([[hip1_hist[end-1], knee1_hist[end-1]]])
            scatter2.set_offsets([[hip2_hist[end-1], knee2_hist[end-1]]])
        return line1, line2, scatter1, scatter2

    anim = FuncAnimation(fig, update_gait, frames=n_frames,
                         init_func=init_gait, blit=True, interval=80)
    path = os.path.join(OUT_DIR, "gait_analysis.gif")
    anim.save(path, writer=PillowWriter(fps=12), dpi=120)
    plt.close(fig)
    print(f"  ✓ Saved {path}")
    return path


def _evaluate(policy, env, device, episodes=3):
    policy.eval()
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = policy.step(st)
            s, r, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            ep_r += r
        returns.append(ep_r)
    policy.train()
    return float(np.mean(returns))


def _eval_metrics(policy, env, device, episodes=2):
    policy.eval()
    speeds, energies, drifts = [], [], []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = policy.step(st)
            s, r, term, trunc, info = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            speeds.append(info["vel_x"])
            energies.append(info["energy_cost"])
            drifts.append(abs(info["hull_angle"]))
    policy.train()
    return (float(np.mean(speeds)), float(np.mean(energies)),
            float(np.mean(drifts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulation visualisations")
    parser.add_argument("--all", action="store_true", help="Run all simulations")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--gait", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.training, args.curriculum, args.metrics, args.gait]):
        args.all = True

    if args.all or args.training:
        simulate_training_progress()
    if args.all or args.curriculum:
        simulate_curriculum_comparison()
    if args.all or args.metrics:
        simulate_performance_metrics()
    if args.all or args.gait:
        simulate_gait_analysis()

    print(f"\n[simulate] All assets saved to {OUT_DIR}/")
