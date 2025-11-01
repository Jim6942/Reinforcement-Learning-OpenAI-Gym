# train_lander_resume.py
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

ENV_ID   = "LunarLander-v3"
SAVE_DIR = "agents/lunar_ppo"
BEST     = os.path.join(SAVE_DIR, "best_model.zip")
LAST     = os.path.join(SAVE_DIR, "last_model.zip")

os.makedirs(SAVE_DIR, exist_ok=True)

def make_env():
    return Monitor(gym.make(ENV_ID))

train_env = make_env()
eval_env  = make_env()

# -------- resume or start fresh --------
if os.path.exists(BEST):
    print(f"[info] Resuming from {BEST}")
    model = PPO.load(BEST, env=train_env, print_system_info=True)
else:
    print("[info] Starting new PPO model")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        seed=42,
        device="auto",
    )

# -------- callbacks: eval + periodic checkpoints --------
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,  # writes/overwrites best_model.zip
    log_path=SAVE_DIR,
    eval_freq=10_000,               # evaluate every 10k steps
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)
ckpt_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=SAVE_DIR,
    name_prefix="ckpt",
    save_replay_buffer=False,
    save_vecnormalize=False,
)
callbacks = CallbackList([eval_cb, ckpt_cb])

# -------- train --------
TOTAL_STEPS = 1_000_000
print(f"[info] Learning for {TOTAL_STEPS:,} steps…")
model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=False)
 
# Save last snapshot too
model.save(LAST)
print("[info] Training complete. Best at:", BEST)
