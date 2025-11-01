import base64, io, os, uuid, math
from typing import Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    import gymnasium as gym
except ImportError:
    import gym
from PIL import Image

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://ailunarlander.netlify.app",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewReq(BaseModel):
    env_id: str = "LunarLander-v3"
    angle_deg: Optional[float] = None
    x_threshold: Optional[float] = None
    max_steps: Optional[int] = None
    seed: Optional[int] = None
    render_w: Optional[int] = None
    render_h: Optional[int] = None

class SidReq(BaseModel):
    session_id: str
    repeat: Optional[int] = 1

class StepReq(SidReq):
    action: int

def load_agent(env_id: str):
    try:
        from stable_baselines3 import PPO, DQN
    except Exception:
        return None
    if "LunarLander" in env_id:
        path = "agents/lunar_ppo/best_model.zip"
        if os.path.exists(path):
            try:
                return PPO.load(path)
            except Exception:
                return None
    if "CartPole" in env_id:
        path = "agents/cartpole_dqn/best_model.zip"
        if os.path.exists(path):
            try:
                return DQN.load(path)
            except Exception:
                return None
    return None

class Session:
    def __init__(self, env_id: str, angle_deg: Optional[float], x_threshold: Optional[float], max_steps: Optional[int], seed: Optional[int], w: Optional[int], h: Optional[int]):
        self.env_id = env_id
        self.seed = seed
        self.rw = int(w) if w else 400
        self.rh = int(h) if h else 300
        kwargs = {"render_mode": "rgb_array"}
        if max_steps is not None:
            kwargs["max_episode_steps"] = int(max_steps)
        self.env = gym.make(env_id, **kwargs)
        unwrapped = self.env.unwrapped
        if angle_deg is not None and hasattr(unwrapped, "theta_threshold_radians"):
            unwrapped.theta_threshold_radians = float(angle_deg) * math.pi / 180.0
        if x_threshold is not None and hasattr(unwrapped, "x_threshold"):
            unwrapped.x_threshold = float(x_threshold)
        self.obs, _ = self.env.reset(seed=seed)
        self.done = False
        self.agent = load_agent(env_id)

    def frame(self) -> str:
        arr = self.env.render()
        if arr is None:
            arr = np.zeros((self.rh, self.rw, 3), dtype=np.uint8)
        im = Image.fromarray(arr)
        im = im.resize((self.rw, self.rh))
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=60, method=3)
        return base64.b64encode(buf.getvalue()).decode("ascii")

SESS: Dict[str, Session] = {}

def get_session(sid: str) -> Session:
    s = SESS.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/session/new")
def new(req: NewReq):
    sid = str(uuid.uuid4())
    SESS[sid] = Session(req.env_id, req.angle_deg, req.x_threshold, req.max_steps, req.seed, req.render_w, req.render_h)
    s = SESS[sid]
    return {"session_id": sid, "obs": s.obs.tolist(), "frame": s.frame(), "done": s.done}

@app.post("/reset")
def reset(req: SidReq):
    s = get_session(req.session_id)
    s.obs, _ = s.env.reset(seed=s.seed)
    s.done = False
    return {"obs": s.obs.tolist(), "frame": s.frame(), "done": s.done}

@app.post("/step")
def step(req: StepReq):
    s = get_session(req.session_id)
    repeat = int(req.repeat or 1)
    if repeat < 1:
        repeat = 1
    total_rew = 0.0
    if s.done:
        return {"obs": s.obs.tolist(), "reward": 0.0, "done": True, "frame": s.frame(), "steps": 0}
    for i in range(repeat):
        out = s.env.step(int(req.action))
        if len(out) == 5:
            obs, rew, term, trunc, _ = out
            done = bool(term or trunc)
        else:
            obs, rew, done, _ = out
        s.obs = obs
        total_rew += float(rew)
        s.done = done
        if s.done:
            break
    return {"obs": s.obs.tolist(), "reward": total_rew, "done": s.done, "frame": s.frame(), "steps": i + 1}

@app.post("/agent_step")
def agent_step(req: SidReq):
    s = get_session(req.session_id)
    if s.agent is None:
        raise HTTPException(400, "No agent loaded for this env.")
    repeat = int(req.repeat or 1)
    if repeat < 1:
        repeat = 1
    total_rew = 0.0
    last_action = 0
    if s.done:
        return {"obs": s.obs.tolist(), "reward": 0.0, "done": True, "frame": s.frame(), "steps": 0}
    for i in range(repeat):
        action, _ = s.agent.predict(s.obs, deterministic=True)
        last_action = int(action)
        obs, rew, term, trunc, _ = s.env.step(last_action)
        s.obs = obs
        total_rew += float(rew)
        s.done = bool(term or trunc)
        if s.done:
            break
    return {"obs": s.obs.tolist(), "reward": total_rew, "done": s.done, "frame": s.frame(), "action": last_action, "steps": i + 1}
